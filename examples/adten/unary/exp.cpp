// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin unary_exp usr}

Exponentiation
##############
{xrst_literal ,
    BEGIN_CPP, END_CPP
}

{xrst_end unary_exp}
*/
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
TEST(examples_adten, unary_exp)  {
    using ad_tensor::adten_t;
    using ad_tensor::vector;
    using at::Tensor;
    //
    // v, f
    vector<Tensor> v;
    v.push_back( torch::tensor( {2.0, 4.0} ) );
    vector<adten_t> av = adten_t::start_recording(v);
    vector<adten_t>  ar;
    ar.push_back( av[0].exp() );
    ad_tensor::adfn_t f  = adten_t::stop_recording(ar, "f");
    //
    // v_all, r
    vector<Tensor> v_all = f.forward_var(v);
    vector<Tensor> r     = f.get_range(v_all);
    EXPECT_TRUE( r[0].equal( v[0].exp() ) );
    //
    // dv, dr
    vector<Tensor> dv;
    dv.push_back( torch::tensor( {3.0, 5.0} ) );
    vector<Tensor> dr = f.forward_der(dv, v_all);
    EXPECT_TRUE( dr[0].equal( r[0] * dv[0] ) );
    //
    // dr, dv
    dr[0] = torch::tensor( {6.0, 7.0} );
    dv     = f.reverse_der(dr, v_all);
    EXPECT_TRUE( dv[0].equal( r[0] * dr[0] ) );
}
// END_CPP
