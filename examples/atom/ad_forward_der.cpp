// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin atom_ad_forward_der usr}
{xrst_spell
    nowrap
    lll
}


Recording Forward Derivatives That Use Atomic Functions
#######################################################

derive_atom_y_t
***************
This example use the atomic function :math:`y(x) = x^3`; see
:ref:`ad_atom_example@derive_atom_y_t` .

.. math::
    :nowrap:

    \begin{array}{lll}
    f(v)     &= {\rm sum} ( y(v) )  &= {\rm sum} ( v^3 ) \\
    g(v)     &= f'(v)               &= {\rm sum} (3 * v^2 ) \\
    g'(v)    &= f^{(2)} (v)         &= {\rm sum} (6 * v ) \\
    \end{array}

where sum above sums the elements of the corresponding tensor.


Source Code
***********
{xrst_literal ,
    BEGIN_CPP, END_CPP
}
{xrst_end atom_ad_forward_der}
*/
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
//
#include "ad_atom.hpp"
TEST(examples_atom, ad_forward_der)  {
    //
    // atom_id_z
    std::unique_ptr<ad_tensor::base_atom_t> base_atom_z_ptr =
        std::make_unique<derive_atom_z_t>();
    size_t atom_id_z = ad_tensor::make_atom(base_atom_z_ptr);
    //
    // atom_id_y
    std::unique_ptr<ad_tensor::base_atom_t> base_atom_y_ptr =
        std::make_unique<derive_atom_y_t>(atom_id_z);
    size_t atom_id_y = ad_tensor::make_atom(base_atom_y_ptr);
    //
    // v, av
    vector<Tensor> v;
    v.push_back( torch::tensor( {0.0, 0.0} ) );
    vector<adten_t> av = adten_t::start_recording(v);
    //
    // ay
    vector<adten_t> ay = ad_tensor::call_atom(atom_id_y, av);
    //
    // ar
    vector<adten_t> ar;
    ar.push_back( ay[0].sum() );
    //
    // r = f(v) = (v * v * v).sum()
    ad_tensor::adfn_t f = adten_t::stop_recording(ar, "f");
    //
    // av, av_all
    av = adten_t::start_recording(v);
    vector<adten_t> av_all = f.forward_var(av);
    //
    // fp = g(v) = f'(v) = (3 * v * v).sum()
    vector<adten_t> adv;
    adv.push_back( adten_t( torch::tensor( {1.0, 1.0} ) ) );
    vector<adten_t>   afp = f.forward_der(adv, av_all);
    ad_tensor::adfn_t g   = adten_t::stop_recording(afp, "g");
    //
    // v, v_all, fp
    v[0] = torch::tensor( {3.0, 4.0} );
    vector<Tensor> v_all = g.forward_var(v);
    vector<Tensor> fp    = g.get_range(v_all);
    //
    // check
    EXPECT_EQ(fp[0].item<float>(), (3.0 * v[0] * v[0]).sum().item<float>() );
    //
    // dv, gp
    vector<Tensor> dv;
    dv.push_back( torch::tensor( {1.0, 1.0} ) );
    vector<Tensor> gp = g.forward_der(dv, v_all);
    //
    // check
    EXPECT_EQ(gp[0].item<float>(), (6.0 * v[0]).sum().item<float>() );
}
// END_CPP
