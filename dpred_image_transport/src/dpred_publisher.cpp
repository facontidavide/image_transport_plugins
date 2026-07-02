// Copyright (c) 2026, Davide Faconti
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the copyright holder nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "dpred_image_transport/dpred_publisher.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include "depth_codec.hpp"

namespace dpred_image_transport
{

DpredPublisher::DpredPublisher()
: logger_(rclcpp::get_logger("DpredPublisher"))
{
}

void DpredPublisher::advertiseImpl(
  image_transport::RequiredInterfaces node_interfaces,
  const std::string & base_topic,
  rclcpp::QoS custom_qos,
  rclcpp::PublisherOptions options)
{
  node_param_interface_ = node_interfaces.get_node_parameters_interface();
  typedef image_transport::SimplePublisherPlugin<CompressedImage> Base;
  Base::advertiseImpl(node_interfaces, base_topic, custom_qos, options);

  // Transport-scoped parameter (e.g. image_raw.dpred.zstd_level).
  const unsigned int ns_len =
    std::string(node_interfaces.get_node_base_interface()->get_namespace()).length();
  std::string param_base_name = base_topic.substr(ns_len);
  std::replace(param_base_name.begin(), param_base_name.end(), '/', '.');
  level_param_name_ = param_base_name + "." + getTransportName() + ".zstd_level";

  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.name = "zstd_level";
  descriptor.type = rcl_interfaces::msg::ParameterType::PARAMETER_INTEGER;
  descriptor.description = "zstd level of the dpred entropy stage (1-3)";
  descriptor.integer_range = {rcl_interfaces::msg::IntegerRange()
    .set__from_value(1)
    .set__to_value(3)
    .set__step(1)};
  try {
    node_param_interface_->declare_parameter(
      level_param_name_, rclcpp::ParameterValue(1), descriptor);
  } catch (const rclcpp::exceptions::ParameterAlreadyDeclaredException &) {
    RCLCPP_DEBUG(logger_, "%s was previously declared", level_param_name_.c_str());
  }
}

void DpredPublisher::publish(
  const sensor_msgs::msg::Image & message,
  const PublisherT & publisher) const
{
  const bool is_32f = message.encoding == sensor_msgs::image_encodings::TYPE_32FC1;
  const bool is_16u = message.encoding == sensor_msgs::image_encodings::TYPE_16UC1;
  if (!is_32f && !is_16u) {
    RCLCPP_ERROR_ONCE(
      logger_, "dpred transport supports only 32FC1 and 16UC1 depth images, got '%s'. "
      "Use compressed or zstd for other encodings.", message.encoding.c_str());
    return;
  }
  if (message.is_bigendian) {
    RCLCPP_ERROR_ONCE(logger_, "dpred transport does not support big-endian images");
    return;
  }

  const int level = static_cast<int>(
    node_param_interface_->get_parameter(level_param_name_).as_int());

  const uint32_t w = message.width;
  const uint32_t h = message.height;
  const size_t bpp = is_32f ? 4 : 2;
  const uint8_t * src = message.data.data();
  std::vector<uint8_t> packed;
  if (message.step != w * bpp) {  // rows are padded: repack contiguously
    if (message.step < w * bpp || message.data.size() < static_cast<size_t>(message.step) * h) {
      RCLCPP_ERROR_ONCE(logger_, "inconsistent image step/size, dropping frame");
      return;
    }
    packed.resize(static_cast<size_t>(w) * h * bpp);
    for (uint32_t y = 0; y < h; ++y) {
      std::memcpy(
        packed.data() + static_cast<size_t>(y) * w * bpp,
        src + static_cast<size_t>(y) * message.step,
        static_cast<size_t>(w) * bpp);
    }
    src = packed.data();
  }

  try {
    auto compressed = std::make_unique<CompressedImage>();
    compressed->header = message.header;
    compressed->format = message.encoding + "; dpred";
    // The encoder writes the blob directly into the message field.
    if (is_32f) {
      depth_codec::encode_depth(
        reinterpret_cast<const float *>(src), w, h, compressed->data, level);
    } else {
      depth_codec::encode_depth16(
        reinterpret_cast<const uint16_t *>(src), w, h, compressed->data, level);
    }
    publisher->publish(std::move(compressed));
  } catch (const std::exception & e) {
    RCLCPP_ERROR(logger_, "dpred encoding failed: %s", e.what());
  }
}

}  // namespace dpred_image_transport
