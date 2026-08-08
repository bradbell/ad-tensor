// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
-------------------------------------------------------------------------------
{xrst_begin unary_inverse usr}

Matrix Inverse
##############
{xrst_literal ,
    BEGIN_CPP, END_CPP
}

{xrst_end unary_inverse}
*/
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
TEST(examples_adten, unary_inverse)  {
    using ad_tensor::adten_t;
    using ad_tensor::vector;
    using at::Tensor;
    //
    // options
    at::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat64);
    //
    // v
    double v0 = 1.0, v1 = 2.0, v2 = 3.0, v3 = 4.0;
    vector<Tensor> v;
    v.push_back( torch::tensor( {
        {v0, v1},
        {v2, v3}
    }, options ) );
    //
    // determinant
    double determinant = v0 * v3 - v1 * v2;
    //
    // inverse
    /* inv(v) = {
        {v3, -v1},
        {-v2, v0}
    } / det(v0, v1, v2, v3)
    */
    Tensor inverse = torch::tensor(
        { {v3, -v1}, {-v2, v0} }, options
    ) / determinant;
    //
    // f
    vector<adten_t> av = adten_t::start_recording(v);
    vector<adten_t>  ar;
    ar.push_back( av[0].inverse() );
    ad_tensor::adfn_t f  = adten_t::stop_recording(ar, "f");
    //
    // v_all, r
    vector<Tensor> v_all = f.forward_var(v);
    vector<Tensor> r     = f.get_range(v_all);
    EXPECT_TRUE( r[0].allclose( inverse ) );
    //
    // jacobian_dv
    /* inv'(v) * dv =
        { {dv3, -dv1}, {-dv2, dv0} } / det(v0, v1, v2, v3)
        -
        { {v3, -v1}, {-v2, v0} } / det(v0, v1, v2, v3)^2
        *
        ( v0 * dv3 + dv0 * v3 - v1 * dv2 - dv1 * v2)
    */
    //
    // dv, dr
    vector<Tensor> dv;
    double dv0 = 5.0, dv1 = 6.0, dv2 = 7.0, dv3 = 8.0;
    dv.push_back( torch::tensor(
        { {dv0, dv1}, {dv2, dv3} }, options
    ) );
    vector<Tensor> dr = f.forward_der(dv, v_all);
    //
    Tensor ten1 = torch::tensor(
        { {dv3, -dv1}, {-dv2, dv0} }, options
    );
    ten1       /= determinant;
    Tensor ten2 = torch::tensor({
        {v3, -v1}, {-v2, v0} }, options
    );
    ten2       /= determinant * determinant;
    ten2       *= v0 * dv3 + dv0 * v3 - v1 * dv2 - dv1 * v2;
    std::cout << "ten1 - ten2" << ten1 - ten2 << "\n";
    std::cout << "dr[0] = " << dr[0] << "\n";
    EXPECT_TRUE( dr[0].allclose( ten1 - ten2  ) );
# if 0
    //
    // dr, dv
    dr[0] = torch::tensor( {6.0, 7.0} );
    dv     = f.reverse_der(dr, v_all);
    EXPECT_TRUE( dv[0].equal( r[0] * dr[0] ) );
#endif
}
// END_CPP
