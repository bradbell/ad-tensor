// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
namespace {
    using at::Tensor;
    using ad_tensor::vector;
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::direction_t;
}
TEST(examples_chkpnt, ad_reverse_der)  {
    //
    // x
    vector<Tensor> x;
    x.push_back( torch::tensor( {3.0, 4.0} ) );
    //
    // ax
    vector<adten_t> ax = adten_t::start_recording(x);
    //
    // ay
    vector<adten_t> ay;
    ay.push_back( ax[0] * ax[0] );
    //
    // y = f(x) = x * x
    adfn_t f = adten_t::stop_recording(ay, "f");
    //
    // chkpnt_id
    vector<direction_t> directions = { direction_t::reverse };
    size_t chkpnt_id  = adfn_t::make_chkpnt(f, x, directions);
    //
    // z = g(x) = f(x) * x = x * x * x
    ax = adten_t::start_recording(x);
    ay = ad_tensor::call_chkpnt(chkpnt_id, ax);
    vector<adten_t> az;
    az.push_back( ay[0] * ax[0] );
    adfn_t g  = adten_t::stop_recording(az, "g");
    //
    // ax, avar_all;;
    ax = adten_t::start_recording(x);
    vector<adten_t> avar_all = g.forward_var(ax);
    //
    // h(x) = g'(x) = 3.0 * x * x
    vector<adten_t> adz;
    adz.push_back( adten_t( torch::tensor( {1.0, 1.0} ) ) );
    vector<adten_t> adx = g.reverse_der(adz, avar_all);
    adfn_t h = adten_t::stop_recording(adx, "h");
    //
    // x, var_all, gp
    x[0]                   = torch::tensor( {3.0, 4.0} );
    vector<Tensor> var_all = h.forward_var(x);
    vector<Tensor> gp      = h.get_range(var_all);
    //
    // equal
    bool equal =  gp[0].equal( 3.0 * x[0] * x[0] );
    EXPECT_TRUE(equal);
    //
    // gpp
    vector<Tensor> dgp;
    dgp.push_back( torch::tensor( {1.0, 1.0} ) );
    vector<Tensor> gpp = h.reverse_der(dgp, var_all);
    //
    // equal
    equal =  gpp[0].equal( 6.0 * x[0] );
    EXPECT_TRUE(equal);
}
