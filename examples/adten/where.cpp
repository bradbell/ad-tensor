// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <torch/torch.h>
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
TEST(examples_adten, where)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using at::Tensor;
    using ad_tensor::vector;
    //
    // x0, x1, x
    double x0  = -4.0;
    double x1  = 4.0;
    vector<Tensor> x;
    x.push_back( torch::tensor( {x0, x1} ) );
    //
    // ax
    vector<adten_t> ax = adten_t::start_recording(x);
    // ax_pos
    adten_t ax_pos = ax[0] > adten_t( torch::tensor( 0.0 ) );
    //
    // ax_sq
    adten_t ax_sq   = ax[0] * ax[0];
    //
    // ay
    vector<adten_t> ay;
    ay.push_back( adten_t::where(ax_pos, ax[0], ax_sq) );
    //
    // y = f(x)
    adfn_t f = adten_t::stop_recording(ay, "f");
    //
    // v_all
    vector<Tensor> v_all = f.forward_var(x);
    //
    // y
    vector<Tensor> y = f.get_range(v_all);
    EXPECT_TRUE( y[0].equal( torch::tensor({x0 * x0, x1}) ) );
    //
    // dx0, dx1, dx
    double dx0  = -4.0;
    double dx1  = 4.0;
    vector<Tensor> dx;
    dx.push_back( torch::tensor( {dx0, dx1} ) );
    vector<Tensor> dy = f.forward_der(dx, v_all);
    EXPECT_TRUE( dy[0].equal( torch::tensor( {2 * x0 * dx0, dx1} ) ) );
}
// END_CPP
