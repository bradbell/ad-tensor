// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin adfn_forward_var_example usr}

Dependent Variable Example
##########################

Discussion
**********
For this example

.. math::

        f(v) = \left[ \begin{array}{c}
            v_0 * v_1 - 1.0 \\
            v_0 + v_1 + 2.0 \\
        \end{array} \right]

Source Code
***********
{xrst_literal ,
    BEGIN_CPP, END_CPP
}

{xrst_end adfn_forward_var_example}
*/
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
TEST(examples_adfn, forward_var)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using at::Tensor;
    using ad_tensor::vector;
    //
    // v
    // We use v for the domain variables
    vector<Tensor> v;
    v.push_back( torch::tensor( {0.0, 0.0} ) );
    v.push_back( torch::tensor( {0.0, 0.0} ) );
    //
    // av
    vector<adten_t> av = adten_t::start_recording(v);
    //
    // ar
    // We use r for the range space.
    vector<adten_t> ar;
    ar.push_back( av[0] * av[1] - adten_t( torch::tensor(1.0) ) );
    ar.push_back( av[0] + av[1] + adten_t( torch::tensor(2.0) ) );
    //
    // r = f(x)
    adfn_t f = adten_t::stop_recording(ar, "f");
    //
    // v
    v[0] = torch::tensor( {5.0, 4.0} );
    v[1] = torch::tensor( {3.0, 2.0} );
    //
    // v_all
    vector<Tensor> v_all = f.forward_var(v);
    //
    bool equal = v_all[0].equal( v[0] );
    EXPECT_TRUE( equal );
    //
    equal = v_all[1].equal( v[1] );
    EXPECT_TRUE( equal );
    //
    //
    // The checks below are not part of the ad_tensor api.
    // They are only meant to give you an idea of what is stored in v_all.
    //
    equal = v_all[2].equal( v[0] * v[1] );
    EXPECT_TRUE( equal );
    //
    equal = v_all[3].equal( v[0] * v[1] - 1.0 );
    EXPECT_TRUE( equal );
    //
    equal = v_all[4].equal( v[0] + v[1] );
    EXPECT_TRUE( equal );
    //
    equal = v_all[5].equal( v[0] + v[1] + 2.0 );
    EXPECT_TRUE( equal );
}
// END_CPP
