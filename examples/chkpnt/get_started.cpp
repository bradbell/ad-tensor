// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/adfn.hpp>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/chkpnt.hpp>
#include <ad_tensor/sparsity.hpp>
#include <ad_tensor/options.hpp>
#include <ad_tensor/vector.hpp>
namespace {
    using at::Tensor;
    using ad_tensor::vector;
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
}
TEST(examples_chkpnt, get_started)  {
    //
    // options
    ad_tensor::options_t options;
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
    size_t chkpnt_id  = ad_tensor::make_chkpnt(f);
    //
    // ax, ay, az
    ax = adten_t::start_recording(x);
    ay = adten_t::call_chkpnt(chkpnt_id, ax);
    vector<adten_t> az;
    az.push_back( ay[0] * ax[0] );
    //
    // z = g(x) = f(x) * x = x * x * x
    adfn_t g  = adten_t::stop_recording(az, "g");
    //
    // x, z
    x[0] = torch::tensor( {3.0, 4.0} );
    vector<Tensor> var_all = g.forward_var(x);
    vector<Tensor> z       = g.get_range(var_all);
    //
    // equal
    bool equal =  z[0].equal( x[0] * x[0] * x[0] );
    EXPECT_TRUE(equal);
    //
    // dx, dz
    vector<Tensor> dx;
    dx.push_back( torch::tensor( {1.0, 2.0} ) );
    vector<Tensor> dz = g.forward_der(dx, var_all);
    //
    // equal
    equal =  dz[0].equal( 3.0 * x[0] * x[0] * dx[0] );
    EXPECT_TRUE(equal);
    //
    // dz, dx
    dz[0] = torch::tensor( {2.0, 1.0} );
    dx = g.reverse_der(dz, var_all);
    //
    // equal
    equal =  dx[0].equal( 3.0 * x[0] * x[0] * dz[0] );
    EXPECT_TRUE(equal);
}
