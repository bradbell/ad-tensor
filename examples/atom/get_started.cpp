// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin atom_get_started usr}
{xrst_spell
    der
    rng
}

Get Started Using Atomic Functions
##################################

For this example, the atomic function is

.. math::

    f(x) &= x * x \\
    g(v) &= f(v).\texttt{sum}() = (v * v).\texttt{sum}()

where sum above sums the elements of the corresponding tensor.

derive_atom_t
*************

constructor
===========
The constructor sets the short name for this atomic function to square.

forward
=======
There is only one domain and range tensor for this atomic function
and the range value is given by f(domain[0])

depend
======
The range value depends on the domain value so (0,0) is in the
depend sparsity pattern.

forward_der
===========
The domain direction derivative is

    f'(domain[0]) * dom_der[0] = 2 * domain[0] * dom_der[0]

reverse_der
===========
The range direction derivative is

    f'(domain[0]) * rng_der[0] = 2 * domain[0] * rng_der[0]

There is no real difference between the domain and range direction derivatives
because both the domain and range have only one tensor.

Source Code
***********
{xrst_literal ,
    BEGIN_CPP, END_CPP
}


{xrst_end atom_get_started}
*/
// BEGIN_CPP
#include <gtest/gtest.h>
#include <ad_tensor/ad_tensor.hpp>
namespace {
    //
    using std::optional;
    using ad_tensor::vector;
    using ad_tensor::adten_t;
    using ad_tensor::base_atom_t;
    using at::Tensor;
    //
    // derive_atom
    class derive_atom_t : public base_atom_t {
    public:
        // constructor
        derive_atom_t(void) {
            set_name("square");
        }
        // depend
        optional<ad_tensor::sparsity_t> depend(void) const override {
            ad_tensor::sparsity_t sparsity;
            sparsity.push_back( {0, 0} );
            //
            std::optional<ad_tensor::sparsity_t> opt = sparsity;
            return opt;
        }
        // forward
        std::optional< vector<Tensor> > forward(
            const vector<bool>&   rng_used ,
            const vector<Tensor>& domain ) const override {
            //
            // range
            vector<Tensor> range;
            range.push_back( domain[0] * domain[0] );
            //
            std::optional< vector<Tensor> > opt = range;
            return opt;
        }
        // forward_der
        std::optional< vector<Tensor> > forward_der(
            const vector<bool>&   rng_used ,
            const vector<Tensor>& domain   ,
            const vector<Tensor>& dom_der ) const override {
            //
            // rng_der
            vector<Tensor> rng_der;
            if( dom_der[0].numel() == 0 ) {
                rng_der.push_back( torch::empty( {0} ) );
            } else {
                rng_der.push_back( 2.0 * domain[0] * dom_der[0] );
            }
            //
            std::optional< vector<Tensor> > opt = rng_der;
            return opt;
        }
        // reverse_der
        std::optional< vector<Tensor> > reverse_der(
            size_t                call_info ,
            const vector<bool>&   rng_used ,
            const vector<Tensor>& domain   ,
            const vector<Tensor>& rng_der ) const override {
            //
            // dom_der
            vector<Tensor> dom_der;
            if( rng_der[0].numel() == 0 ) {
                dom_der.push_back( torch::empty( {0} ) );
            } else {
                dom_der.push_back( 2.0 * domain[0] * rng_der[0] );
            }
            //
            std::optional< vector<Tensor> > opt = dom_der;
            return opt;
        }
    };
    //
    // base_atom_ptr
    std::unique_ptr<base_atom_t> base_atom_ptr =
        std::make_unique<derive_atom_t>();
}
TEST(examples_atom, get_started)  {
    //
    // atom_id
    size_t atom_id = ad_tensor::make_atom(base_atom_ptr);
    //
    // v
    // We use x for the adfn domain variables
    vector<Tensor> v;
    v.push_back( torch::tensor( {0.0, 0.0} ) );
    //
    // av
    vector<adten_t> av = adten_t::start_recording(v);
    //
    // ar
    vector<adten_t> ay = ad_tensor::call_atom(atom_id, av);
    vector<adten_t> ar;
    ar.push_back( ay[0].sum() );
    //
    // r = g(x) = f(x).sum()
    ad_tensor::adfn_t g = adten_t::stop_recording(ar, "g");
    //
    // v
    v[0] = torch:: tensor( {3.0, 4.0} );
    //
    // r
    vector<Tensor> v_all = g.forward_var(v);
    vector<Tensor> r     = g.get_range(v_all);
    //
    // check
    EXPECT_EQ( r[0].item<float>(), (v[0] * v[0]).sum().item<float>() );
    //
    // dv, dr
    // domain direction derivative
    vector<Tensor> dv;
    dv.push_back( torch::tensor( {5.0, 6.0} ) );
    vector<Tensor> dr = g.forward_der(dv, v_all);
    //
    // check
    EXPECT_EQ( dr[0].item<float>(), (2.0 * v[0] * dv[0]).sum().item<float>() );
    //
    // dr, dv
    // range direction derivative
    dr[0] = torch::tensor( 6.0 );
    dv = g.reverse_der(dr, v_all);
    //
    // check
    bool equal;
    equal = dv[0].equal( dr[0] * 2.0 * v[0] );
    EXPECT_TRUE( equal );
}
// END_CPP
