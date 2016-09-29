#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>


#include "caffe/layer.hpp"
#include "caffe/layers/crop_layer.hpp"
#include "caffe/net.hpp"


namespace caffe {

// --[ an add for HED
template <typename Dtype>
void CropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Construct a map from top blobs to layer inds, skipping over in-place
  // connections.
  map<Blob<Dtype>*, int> down_map;
  for (int layer_ind = 0; layer_ind < this->net_->top_vecs().size();
       ++layer_ind) {
    vector<Blob<Dtype>*> tops = this->net_->top_vecs()[layer_ind];
    for (int top_ind = 0; top_ind < tops.size(); ++top_ind) {
      if (down_map.find(tops[top_ind]) == down_map.end()) {
        down_map[tops[top_ind]] = layer_ind;
      }
    }
  }
  // Walk back from the first bottom, keeping track of all the blobs we pass.
  set<Blob<Dtype>*> path_blobs;
  Blob<Dtype>* blob = bottom[0];
  int layer_ind;
  // TODO this logic can be simplified if all blobs are tops
  path_blobs.insert(blob);
  while (down_map.find(blob) != down_map.end()) {
    layer_ind = down_map[blob];
    if (this->net_->bottom_vecs()[layer_ind].size() == 0) {
      break;
    }
    blob = this->net_->bottom_vecs()[layer_ind][0];
    path_blobs.insert(blob);
  }
  // Now walk back from the second bottom, until we find a blob of intersection.
  Blob<Dtype>* inter_blob = bottom[1];
  while (path_blobs.find(inter_blob) == path_blobs.end()) {
    CHECK(down_map.find(inter_blob) != down_map.end())
        << "Cannot align apparently disconnected blobs.";
    layer_ind = down_map[inter_blob];
    CHECK_GT(this->net_->bottom_vecs()[layer_ind].size(), 0)
        << "Cannot align apparently disconnected blobs.";
    inter_blob = this->net_->bottom_vecs()[layer_ind][0];
  }
  // Compute the coord map from the blob of intersection to each bottom.
  vector<DiagonalAffineMap<Dtype> > coord_maps(2,
      DiagonalAffineMap<Dtype>::identity(2));
  for (int i = 0; i < 2; ++i) {
    for (Blob<Dtype>* blob = bottom[i]; blob != inter_blob;
         blob = this->net_->bottom_vecs()[down_map[blob]][0]) {
      shared_ptr<Layer<Dtype> > layer = this->net_->layers()[down_map[blob]];
      coord_maps[i] = coord_maps[i].compose(layer->coord_map());
    }
  }
  // Compute the mapping from first bottom coordinates to second.
  DiagonalAffineMap<Dtype> crop_map =
      coord_maps[1].compose(coord_maps[0].inv());
  for (int i = 0; i < 2; ++i) {
    // Check for scale mismatch (unfortunately, CHECK_DOUBLE_EQ does not
    // support a message like the other CHECKs).
    CHECK_DOUBLE_EQ(crop_map.coefs()[i].first, 1);
    CHECK_LE(crop_map.coefs()[i].second, 0) << "Negative crop width.";
    // Check that the crop width is an integer.
    CHECK_DOUBLE_EQ(crop_map.coefs()[i].second,
        round(crop_map.coefs()[i].second));
  }
  crop_h_ = - round(crop_map.coefs()[0].second);
  crop_w_ = - round(crop_map.coefs()[1].second);
}

template <typename Dtype>
void CropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[1]->height(),
      bottom[1]->width());
}

template <typename Dtype>
void CropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < top[0]->num(); ++n) {
    for (int c = 0; c < top[0]->channels(); ++c) {
      for (int h = 0; h < top[0]->height(); ++h) {
        caffe_copy(top[0]->width(),
            bottom_data + bottom[0]->offset(n, c, crop_h_ + h, crop_w_),
            top_data + top[0]->offset(n, c, h));
      }
    }
  }
}

template <typename Dtype>
void CropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  if (propagate_down[0]) {
    caffe_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < top[0]->channels(); ++c) {
        for (int h = 0; h < top[0]->height(); ++h) {
          caffe_copy(top[0]->width(),
              top_diff + top[0]->offset(n, c, h),
              bottom_diff + bottom[0]->offset(n, c, crop_h_ + h, crop_w_));
        }
      }
    }
  }
}
// --[ end HED

// --[ an delete for HED
/*
template <typename Dtype>
void CropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // LayerSetup() handles the number of dimensions; Reshape() handles the sizes.
  // bottom[0] supplies the data
  // bottom[1] supplies the size
  const CropParameter& param = this->layer_param_.crop_param();
  CHECK_EQ(bottom.size(), 2) << "Wrong number of bottom blobs.";
  int input_dim = bottom[0]->num_axes();
  const int start_axis = bottom[0]->CanonicalAxisIndex(param.axis());
  CHECK_LT(start_axis, input_dim) << "crop axis bigger than input dim";
  if (param.offset_size() > 1) {
    // the number of crop values specified must be equal to the number
    // of dimensions following axis
    CHECK_EQ(start_axis + param.offset_size(), input_dim)
      << "number of offset values specified must be equal to the number of "
      << "dimensions following axis.";
  }
}

template <typename Dtype>
void CropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const CropParameter& param = this->layer_param_.crop_param();
  int input_dim = bottom[0]->num_axes();
  const int start_axis = bottom[0]->CanonicalAxisIndex(param.axis());

  // Initialize offsets to 0 and the new shape to the current shape of the data.
  offsets = vector<int>(input_dim, 0);
  vector<int> new_shape(bottom[0]->shape());

  // Determine crop offsets and the new shape post-crop.
  for (int i = 0; i < input_dim; ++i) {
    int crop_offset = 0;
    int new_size = bottom[0]->shape(i);
    if (i >= start_axis) {
      new_size = bottom[1]->shape(i);
      if (param.offset_size() == 1) {
        // If only one offset is given, all crops have the same offset.
        crop_offset = param.offset(0);
      } else if (param.offset_size() > 1) {
        // For several offsets, the number of offsets must be equal to the
        // number of dimensions to crop, that is dimensions after the axis.
        crop_offset = param.offset(i - start_axis);
      }
      // Check that the crop and offset are within the dimension's bounds.
      CHECK_GE(bottom[0]->shape(i) - crop_offset, bottom[1]->shape(i))
          << "the crop for dimension " << i << " is out-of-bounds with "
          << "size " << bottom[1]->shape(i) << " and offset " << crop_offset;
    }
    new_shape[i] = new_size;
    offsets[i] = crop_offset;
  }
  top[0]->Reshape(new_shape);
}

template <typename Dtype>
void CropLayer<Dtype>::crop_copy(const vector<Blob<Dtype>*>& bottom,
             const vector<Blob<Dtype>*>& top,
             const vector<int>& offsets,
             vector<int> indices,
             int cur_dim,
             const Dtype* src_data,
             Dtype* dest_data,
             bool is_forward) {
  if (cur_dim + 1 < top[0]->num_axes()) {
    // We are not yet at the final dimension, call copy recursively
    for (int i = 0; i < top[0]->shape(cur_dim); ++i) {
      indices[cur_dim] = i;
      crop_copy(bottom, top, offsets, indices, cur_dim+1,
                src_data, dest_data, is_forward);
    }
  } else {
    // We are at the last dimensions, which is stored continously in memory
    for (int i = 0; i < top[0]->shape(cur_dim); ++i) {
      // prepare index vector reduced(red) and with offsets(off)
      std::vector<int> ind_red(cur_dim, 0);
      std::vector<int> ind_off(cur_dim+1, 0);
      for (int j = 0; j < cur_dim; ++j) {
          ind_red[j] = indices[j];
          ind_off[j] = indices[j] + offsets[j];
      }
      ind_off[cur_dim] = offsets[cur_dim];
      // do the copy
      if (is_forward) {
        caffe_copy(top[0]->shape(cur_dim),
            src_data + bottom[0]->offset(ind_off),
            dest_data + top[0]->offset(ind_red));
      } else {
        // in the backwards pass the src_data is top_diff
        // and the dest_data is bottom_diff
        caffe_copy(top[0]->shape(cur_dim),
            src_data + top[0]->offset(ind_red),
            dest_data + bottom[0]->offset(ind_off));
      }
    }
  }
}

template <typename Dtype>
void CropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  std::vector<int> indices(top[0]->num_axes(), 0);
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  crop_copy(bottom, top, offsets, indices, 0, bottom_data, top_data, true);
}

template <typename Dtype>
void CropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  if (propagate_down[0]) {
    caffe_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
    std::vector<int> indices(top[0]->num_axes(), 0);
    crop_copy(bottom, top, offsets, indices, 0, top_diff, bottom_diff, false);
  }
}
*/


#ifdef CPU_ONLY
STUB_GPU(CropLayer);
#endif

INSTANTIATE_CLASS(CropLayer);
REGISTER_LAYER_CLASS(Crop);

}  // namespace caffe
