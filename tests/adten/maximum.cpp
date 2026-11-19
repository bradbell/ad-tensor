// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/adten.hpp>
//
TEST(tests_adten, maximum)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::vector;
    //
    // first, second
    at::Tensor first  = torch::tensor( {1.0, 2.0, 3.0} );
    at::Tensor second = torch::tensor( {4.0, 2.0, 0.0} );
    at::Tensor third  = torch::tensor( {2.5} );
    //
    // ---------------------------------------------------------------------
    // f
    // ---------------------------------------------------------------------
    // x
    vector<at::Tensor> x = {first, second, third};
    //
    // ax
    vector<adten_t> ax = adten_t::start_recording(x);
    //
    // ay
    vector<adten_t> ay;
    ay.push_back( ax[0].maximum( ax[1] ) );
    ay.push_back( ax[1].maximum( ax[2] ) );
    //
    // y = f(x)
    adfn_t f = adten_t::stop_recording(ay, "f");
    //
    // v_all
    vector<at::Tensor> v_all = f.forward_var(x);
    // ---------------------------------------------------------------------
    // y
    // ---------------------------------------------------------------------
    vector<at::Tensor> y = f.get_range(v_all);
    EXPECT_EQ( y.size(), ay.size() );
    //
    at::Tensor check;
    check = torch::tensor( {4.0, 2.0, 3.0} );
    EXPECT_TRUE( y[0].equal( check ) );
    //
    check = torch::tensor( {4.0, 2.5, 2.5} );
    EXPECT_TRUE( y[1].equal( check ) );
    // ---------------------------------------------------------------------
    // dx, dy
    // ---------------------------------------------------------------------
    at::Tensor         first_dx   = torch::tensor( {5.0, 6.0, 7.0} );
    at::Tensor         second_dx  = torch::tensor( {8.0, 6.0, 9.0} );
    at::Tensor         third_dx   = torch::tensor( {10.0} );
    vector<at::Tensor> dx         = {first_dx, second_dx, third_dx};
    vector<at::Tensor> dy         = f.forward_der(dx, v_all);
    //
    check                         = torch::tensor( {8.0, 6.0, 7.0} );
    EXPECT_TRUE( dy[0].equal(check) );
    //
    check                         = torch::tensor( {8.0, 10.0, 10.0} );
    EXPECT_TRUE( dy[1].equal(check) );
    // ---------------------------------------------------------------------
    // py, px
    // ---------------------------------------------------------------------
    at::Tensor         first_py   = torch::tensor( {5.0, 0.0, 7.0} );
    at::Tensor         second_py  = torch::tensor( {8.0, 0.0, 9.0} );
    vector<at::Tensor> py         = {first_py, second_py};
    vector<at::Tensor> px         = f.reverse_der(py, v_all);
    //
    check                         = torch::tensor( {0.0, 0.0, 7.0} );
    EXPECT_TRUE( px[0].equal(check) );
    //
    check                         = torch::tensor( {5.0 + 8.0, 0.0, 0.0} );
    EXPECT_TRUE( px[1].equal(check) );
    //
    check                         = torch::tensor( {9.0} );
    EXPECT_TRUE( px[2].equal(check) );
}
// END_CPP
