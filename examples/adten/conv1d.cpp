// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin example_conv1d usr}

Examples One Dimensional Cross Correlation
##########################################


{xrst_end example_conv1d}
*/
// BEGIN_CPP
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/ad_tensor.hpp>
TEST(examples_adten, conv1d) {
    using ad_tensor::adten_t;
    using ad_tensor::vector;
    //
    // input_vec, weight_vec, bias_vec
    vector<float> input_vec  = {1.0, 2.0, 3.0, 4.0};
    vector<float> weight_vec = {0.25, 0.5, 0.25};
    vector<float> bias_vec   = {-1.0};
    //
    // input, weight, bias
    at::Tensor input  = torch::tensor(input_vec).view( {1, 1, 4} );
    at::Tensor weight = torch::tensor(weight_vec).view( {1, 1, 3} );
    at::Tensor bias   = torch::tensor(bias_vec);
    //
    // ainput, aweight, abias
    adten_t ainput  = adten_t(input);
    adten_t aweight = adten_t(weight);
    adten_t abias   = adten_t(bias);
    //
    // output
    auto options   = torch::nn::functional::Conv1dFuncOptions();
    adten_t   aoutput = ad_tensor::conv1d(ainput, aweight, abias, options);
    at::Tensor output = aoutput.at_ten().contiguous();
    //
    std::cout << "output.sizes() = " << output.sizes() << "\n";
    std::cout << "output = " << output << "\n";
    //
    // n_input_cross, n_weight_cross, n_output_cross
    int64_t n_input_cross  = input.sizes()[2];
    int64_t n_weight_cross = weight.sizes()[2];
    int64_t n_output_cross = output.sizes()[2];
    EXPECT_EQ( n_output_cross,  n_input_cross - n_weight_cross + 1 );
    //
    // output_vec
    vector<float> output_vec(
        output.data_ptr<float>(),
        output.data_ptr<float>() + output.numel()
    );
    //
    for(int64_t i = 0; i < n_output_cross; ++i) {
        float sum = bias_vec[0];
        for(int64_t j = 0; j < n_weight_cross; ++j) {
            sum += input_vec[i + j] * weight_vec[j];
        }
        EXPECT_EQ( output_vec[i], sum );
    }
}
// END_CPP
