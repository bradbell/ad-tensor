// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin adfn_get_range_example usr}

Get Ranger Tensor Example
#########################


Discussion
**********
For this example

.. math::

        f(v, p) = \left[ \begin{array}{c}
            c \\
            p_0 + c \\
            v_0 + c \\
        \end{array} \right]

Source Code
***********
{xrst_literal ,
    BEGIN_CPP, END_CPP
}

{xrst_end adfn_get_range_example}
*/
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
TEST(examples_adfn, get_range)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using at::Tensor;
    using ad_tensor::vector;
    //
    // c
    Tensor c = torch::tensor( {1.0, 2.0} );
    //
    // p
    vector<Tensor> p;
    p.push_back( torch::tensor( {0.0, 0.0} ) );
    //
    // v
    vector<Tensor> v;
    v.push_back( torch::tensor( {0.0, 0.0} ) );
    //
    // av, ap
    auto [av, ap] = adten_t::start_recording(v, p);
    //
    // ac
    adten_t ac = adten_t( c );
    //
    // create a parameter and variable that are not used
    ap[0] - ac;
    av[0] / ac;
    //
    // ar
    vector<adten_t> ar;
    ar.push_back( ac );            // a constant
    ar.push_back( ap[0] + ac );    // a parameter
    ar.push_back( av[0] + ac );    // a variable
    //
    // r = f(v, p)
    adfn_t f = adten_t::stop_recording(ar, "f");
    //
    // p, v
    p[0] = torch::tensor( {7.0, 8.0} );
    v[0] = torch::tensor( {9.0, 10.0} );
    //
    // r
    vector<Tensor> p_all = f.forward_par(p);
    vector<Tensor> v_all = f.forward_var(v, p_all);
    vector<Tensor> r     = f.get_range(v_all, p_all);
    //
    EXPECT_EQ( r.size(), 3 );
    //
    bool equal = r[0].equal( c );
    EXPECT_TRUE( equal );
    //
    equal = r[1].equal( p[0] + c );
    EXPECT_TRUE( equal );
    //
    equal = r[2].equal( v[0] + c );
    EXPECT_TRUE( equal );
}
// END_CPP
