# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import tvm
from tvm.relax.backend.contrib.dnnl import partition_for_dnnl
from tvm.script import relax as R


# 1. Test the Safety Guards (int64 rejection)
@tvm.script.ir_module
class Int64Conv:
    @R.function
    def main(
        data: R.Tensor((1, 3, 224, 224), dtype="int64"),
        weight: R.Tensor((16, 3, 3, 3), dtype="int64"),
    ):
        with R.dataflow():
            # This should NOT be offloaded to DNNL because of our int64 checker
            conv = R.nn.conv2d(data, weight, padding=(1, 1))
            R.output(conv)
        return conv


# 2. Test the ResNet Stride-2 Rewrite
@tvm.script.ir_module
class ResNetBlock:
    @R.function
    def main(
        data: R.Tensor((1, 256, 56, 56), dtype="float32"),
        w_1x1: R.Tensor((64, 256, 1, 1), dtype="float32"),
        w_3x3: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ):
        with R.dataflow():
            # 1x1 conv starts with stride=2 (the ResNet-v1 default)
            conv1 = R.nn.conv2d(data, w_1x1, strides=(2, 2))
            relu1 = R.nn.relu(conv1)
            # 3x3 conv has stride=1
            conv2 = R.nn.conv2d(relu1, w_3x3, strides=(1, 1), padding=(1, 1))
            R.output(conv2)
        return conv2


if __name__ == "__main__":
    print("=== Testing int64 Safety Guard ===")
    mod_int64 = partition_for_dnnl(Int64Conv)
    # If the checker works, 'relax.nn.conv2d' will remain in the main function.
    # If it fails, it will be replaced by a call to a 'dnnl_tensor_...' GlobalVar.
    print(mod_int64.script())

    print("\n=== Testing ResNet Downsample Rewrite ===")
    mod_resnet = partition_for_dnnl(ResNetBlock)
    # You should see the strides swapped in the partitioned graph's composite functions:
    # The 1x1 (first op) should now have strides=[1, 1]
    # The 3x3 (second op) should now have strides=[2, 2]
    print(mod_resnet.script())
