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
TEST(examples_chkpnt, ad_forward_der)  {
    //
    // x
    vector<Tensor> x;
    x.push_back( torch::tensor( {1.0, 2.0} ) );
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
    vector<direction_t> directions = { direction_t::forward };
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
    // adz = h(x) = f'(x) = 3.0 * x * x * x
    vector<adten_t> adx;
    adx.push_back( adten_t( torch::tensor( {1.0, 1.0} ) ) );
    vector<adten_t> adz = g.forward_der(adx, avar_all);
    adfn_t h = adten_t::stop_recording(adz, "h");
    //
    // x, var_all, dz
    x[0]                   = torch::tensor( {3.0, 4.0} );
    vector<Tensor> var_all = h.forward_var(x);
    vector<Tensor> dz      = h.get_range(var_all);
    //
    // equal
    bool equal =  dz[0].equal( 3.0 * x[0] * x[0] );
    EXPECT_TRUE(equal);
    //
    // ddz
    vector<Tensor> dx;
    dx.push_back( torch::tensor( {1.0, 1.0} ) );
    vector<Tensor> ddz = h.forward_der(dx, var_all);
    //
    // equal
    equal =  ddz[0].equal( 6.0 * x[0] );
    EXPECT_TRUE(equal);
}
