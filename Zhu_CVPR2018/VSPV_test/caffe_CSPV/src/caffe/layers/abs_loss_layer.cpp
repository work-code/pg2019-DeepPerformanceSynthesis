#include <vector>
#include <cmath>

#include "caffe/layers/abs_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AbsLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if ( has_ignore_label_ && !this->layer_param_.loss_param().has_ignore_mode()){
    ignore_mode_ = LossParameter_IgnoreMode_VALUE; 
  }
  else {
    ignore_mode_ = this->layer_param_.loss_param().ignore_mode();
  }
}

template <typename Dtype>
void AbsLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Dtype loss = 0;
  Dtype* diff__data = diff_.mutable_cpu_data();
  for (int i=0; i<count; ++i){
      if (has_ignore_label_) {
          const Dtype* label_data = bottom[1]->cpu_data();
          const int label_value = static_cast<int>(label_data[i]);
          switch (ignore_mode_) {
             case LossParameter_IgnoreMode_VALUE:
              if (label_value == ignore_label_) {
                 diff__data[i] = 0;
              }
              break;
             case LossParameter_IgnoreMode_THRES:
                if (label_value >= ignore_label_) {
                   diff__data[i] = 0;
                }
             break;
             default:
               LOG(FATAL) << "Unknown ignore mode: "
                   << LossParameter_IgnoreMode_Name(ignore_mode_);
          }
      }
      loss = loss + std::abs(diff__data[i]);
  }
  //Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  loss = loss / bottom[0]->num();
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void AbsLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
      const Dtype* diff__data = diff_.cpu_data();
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      int count =  bottom[i] -> count();
      for (int j = 0; j < count; ++j){
          if ( diff__data[j]>0 ) {
	     bottom_diff[j] = alpha;
	  }
	  else if ( diff__data[j]<0 ) {
             bottom_diff[j] = -alpha;
	  }
	  else {
  	     bottom_diff[j] = Dtype(0);
	  }
      }
//      caffe_cpu_axpby(
//          bottom[i]->count(),              // count
//          alpha,                              // alpha
//          diff_.cpu_data(),                   // a
//          Dtype(0),                           // beta
//          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(AbsLossLayer);
#endif

INSTANTIATE_CLASS(AbsLossLayer);
REGISTER_LAYER_CLASS(AbsLoss);

}  // namespace caffe
