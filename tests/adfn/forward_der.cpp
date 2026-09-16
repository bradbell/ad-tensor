// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/no_elements.hpp>
//
TEST(tests_adfn, forward_der_no_elements) {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using ad_tensor::vector;
    using ad_tensor::no_elements;
    //
    // nv
    size_t nv = 2;
    //
    // v
    vector<at::Tensor> v;
    for(size_t j = 0; j < nv; ++j) {
        v.push_back( torch::tensor( { double(j) } ) );
    }
    //
    // av
    vector<adten_t> av = adten_t::start_recording(v);
    //
    vector<adten_t> ar;
    for(size_t j = 0; j < nv; ++j) {
        ar.push_back( av[j] * av[j] );
    }
    // r = f(vp)
    adfn_t f = adten_t::stop_recording(ar, "f");
    //
    // v_all
    vector<at::Tensor> v_all = f.forward_var(v);
    //
    // dv, dr
    vector<at::Tensor> dv, dr;
    for(size_t j = 0; j < nv; ++j) {
        dv.push_back( no_elements() );
    }
    //
    // check
    for(size_t j = 0; j < nv; ++j) {
        dv[j] = torch::tensor( { 1.0 } );
        dr    = f.forward_der(dv, v_all);
        dv[j] = no_elements();
        //
        for(size_t k = 0; k < nv; ++k) {
            if( k == j ) {
                EXPECT_EQ( dr[k].item<double>(), 2.0 * v[k].item<double>() );
            } else {
                EXPECT_TRUE( no_elements( dr[k] ) );
            }
        }
    }
}
