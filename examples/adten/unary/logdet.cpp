// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin unary_logdet usr}

Example Log Of Determinant
##########################
{xrst_literal ,
    BEGIN_CPP, END_CPP
}

{xrst_end unary_logdet}
*/
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
TEST(examples_adten, unary_logdet)  {
    using ad_tensor::adten_t;
    using ad_tensor::vector;
    using at::Tensor;
    //
    // v0, v1, v2, v3
    double v0 = 2.0, v1 = 1.0, v2= 4.0, v3 = 3.0;
    //
    // det
    double det = v0 * v3 - v1 * v2;
    //
    // derivative_logdet
    Tensor derivative_logdet = torch::tensor({ {v3, -v2}, {-v1, v0} }) / det;
    //
    // v, f
    vector<Tensor> v;
    v.push_back( torch::tensor( { {v0, v1}, {v2, v3} } ) );
    vector<adten_t> av = adten_t::start_recording(v);
    vector<adten_t>  ar;
    ar.push_back( av[0].logdet() );
    ad_tensor::adfn_t f  = adten_t::stop_recording(ar, "f");
    //
    // v_all, r
    vector<Tensor> v_all = f.forward_var(v);
    vector<Tensor> r     = f.get_range(v_all);
    //
    // check
    Tensor check = torch::tensor( std::log( det ) );
    EXPECT_TRUE( r[0].allclose( check ) );
    //
    // dv, dr
    vector<Tensor> dv;
    dv.push_back( torch::tensor( { {5.0, 6.0}, {7.0, 8.0} } ) );
    vector<Tensor> dr = f.forward_der(dv, v_all);
    //
    // check
    check = (derivative_logdet * dv[0]).sum();
    EXPECT_TRUE( dr[0].equal( check ) );
    //
    // dr, dv
    dr[0] = torch::tensor( 9.0 );
    dv     = f.reverse_der(dr, v_all);
    //
    // check
    check = dr[0] * derivative_logdet;
    EXPECT_TRUE( dv[0].equal( check ) );
}
// END_CPP
