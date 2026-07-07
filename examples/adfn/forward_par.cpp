// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin adfn_forward_par_example usr}

Dependent Parameter Example
###########################

Discussion
**********
For this example

.. math::

        f(p) = \left[ \begin{array}{c}
            p_0 * p_1 - 1.0 \\
            p_0 + p_1 + 2.0 \\
        \end{array} \right]

Source Code
***********
{xrst_literal ,
    BEGIN_CPP, END_CPP
}

{xrst_end adfn_forward_par_example}
*/
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
TEST(examples_adfn, forward_par)  {
    using ad_tensor::adten_t;
    using ad_tensor::adfn_t;
    using at::Tensor;
    using ad_tensor::vector;
    //
    // p
    // We use p for the domain parameters
    vector<Tensor> p;
    p.push_back( torch::ones( {2} ) );
    p.push_back( torch::zeros( {2} ) );
    //
    // ap
    vector<Tensor> v;
    auto [av, ap] = adten_t::start_recording(v, p);
    //
    // ar
    vector<adten_t> ar;
    ar.push_back( ap[0] * ap[1] - adten_t( torch::tensor(1.0) ) );
    ar.push_back( ap[0] + ap[1] + adten_t( torch::tensor(2.0) ) );
    //
    // r = f(p)
    adfn_t f = adten_t::stop_recording(ar, "f");
    //
    // p
    p[0] = torch::tensor( {1.0, 2.0} );
    p[1] = torch::tensor( {3.0, 4.0} );
    //
    // p_all
    vector<Tensor> p_all = f.forward_par(p);
    //
    bool equal = p_all[0].equal( p[0] );
    EXPECT_TRUE( equal );
    //
    equal = p_all[1].equal( p[1] );
    EXPECT_TRUE( equal );
    //
    // The checks below are not part of the ad_tensor api
    // and may change in the future. They are only meant to give you an idea
    // of what is stored in p_all.
    //
    equal = p_all[2].equal( p[0] * p[1] );
    EXPECT_TRUE( equal );
    //
    equal = p_all[3].equal( p[0] * p[1] - 1.0 );
    EXPECT_TRUE( equal );
    //
    equal = p_all[4].equal( p[0] + p[1] );
    EXPECT_TRUE( equal );
    //
    equal = p_all[5].equal( p[0] + p[1] + 2.0 );
    EXPECT_TRUE( equal );
}
// END_CPP
