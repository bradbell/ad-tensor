#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin ad_atom_example usr}
{xrst_spell
    dx
    adten
    namespace
}


Example Atomic Function That Can Record Derivatives
###################################################

Overview
********
This include file defines the atomic functions y and z.
The function y uses z when recording forward or reverse derivatives.

Tracing
*******
If :ref:`atom_non_virtual@trace` is true,
these derived classes print some information about their internal calculations.

derive_atom_y_t
***************
This derived atomic class is defined in the empty namespace.
It implements the following function and domain direction derivative
for vector<at::Tensor> arguments:

.. math::

    y(x)        &= x^3 \\
    y'(x) * dx  &= 3 * x^2 * dx \\

In addition, it uses the z(x, u) atomic function to implement the
domain and range direction derivatives when the argument
type is vector<adten_t>.

.. math::

    y'(x) * dx  &= 3 * x^2 * dx \\
    dy * y'(x)  &= 3 * x^2 * dy \\

derive_atom_z_t
***************
This derived atomic class is defined in the empty namespace.
It implements the following function and domain direction derivative
for vector<at::Tensor> arguments:

.. math::

    z(x, u)                          &= 3 * x^2 * u  \\
    z_x(x, u) * dx + z_u(x, u) * du  &= 6 * x * u * dx + 3 * x^2 * du

using
*****
The following using statements are placed in the empty namespace:
{xrst_literal ,
    BEGIN_USING,  END_USING
}

Example Uses
************
{xrst_toc_table
    examples/atom/ad_forward_der.cpp
    examples/atom/ad_reverse_der.cpp
}

Source Code
***********
{xrst_literal ,
    BEGIN_CPP, END_CPP
}
{xrst_end ad_atom_example}
*/
// BEGIN_CPP
#include <ad_tensor/ad_tensor.hpp>
namespace {
    //
    // BEGIN_USING
    using ad_tensor::vector;
    using ad_tensor::adten_t;
    using at::Tensor;
    using std::cout;
    // END_USING
    //
    // ----------------------------------------------------------------------
    // y(x) = x * x * x
    // ----------------------------------------------------------------------
    //
    // derive_atom_y_t
    class derive_atom_y_t : public ad_tensor::base_atom_t {
    private:
        size_t m_atom_id_z;
    public:
        // ctor
        derive_atom_y_t(size_t atom_id_z) {
            m_atom_id_z = atom_id_z;
            set_name("y");
        }
        // depend
        std::optional<ad_tensor::sparsity_t> depend(
            size_t           call_info) const override {
            ad_tensor::sparsity_t sparsity;
            sparsity.push_back( {0, 0} );
            //
            std::optional<ad_tensor::sparsity_t> opt = sparsity;
            return opt;
        }
        // forward
        std::optional< vector<Tensor> > forward(
            size_t                call_info ,
            const vector<bool>&   rng_used ,
            const vector<Tensor>& domain ) const override {
            //
            Tensor x = domain[0];
            //
            // range
            vector<Tensor> range;
            range.push_back( x * x * x );
            //
            if( get_trace() ) {
                cout << "forward_y: domain =\n" << to_string(domain);
                cout << "forward_y: range =\n" << to_string(range);
            }
            std::optional< vector<Tensor> > opt = range;
            return opt;
        }
        // forward_der
        std::optional< vector<Tensor> > forward_der(
            size_t                call_info ,
            const vector<bool>&   rng_used  ,
            const vector<Tensor>& domain    ,
            const vector<Tensor>& dom_der   ) const override {
            //
            Tensor x  = domain[0];
            Tensor dx = dom_der[0];
            //
            // range
            vector<Tensor> rng_der;
            rng_der.push_back( 3.0 * x * x * dx );
            //
            if( get_trace() ) {
                cout << "forward_der_y: domain =\n" << to_string(domain);
                cout << "forward_der_y: dom_der =\n" << to_string(dom_der);
                cout << "forward_der_y: rng_der =\n" << to_string(rng_der);
            }
            std::optional< vector<Tensor> > opt = rng_der;
            return opt;
        }
        // AD forward_der
        // This is used when recording domain direction derivatives
        std::optional< vector<adten_t> > forward_der(
            size_t                 call_info ,
            const vector<bool>&    rng_used ,
            const vector<adten_t>& domain   ,
            const vector<adten_t>& dom_der ) const override {
            //
            // rng_der
            vector<adten_t> rng_der;
            //
            // rng_der
            if( dom_der[0].numel() == 0 ) {
                rng_der.push_back( adten_t( torch::empty( {0} ) ) );
            } else {
                vector<adten_t> domain_z;
                domain_z.push_back( domain[0] );
                domain_z.push_back( dom_der[0] );
                //
                rng_der = ad_tensor::call_atom(m_atom_id_z, domain_z);
            }
            if( get_trace() ) {
                cout << "AD forward_der_y: domain =\n" << to_string(domain);
                cout << "AD forward_der_y: dom_der =\n" << to_string(dom_der);
                cout << "AD forward_der_y: rng_der =\n" << to_string(rng_der);
            }
            // opt
            std::optional< vector<adten_t> > opt = rng_der;
            return opt;
        }
        // AD reverse_der
        // This is used when recording range direction derivatives
        std::optional< vector<adten_t> > reverse_der(
            size_t                 call_info ,
            const vector<bool>&    rng_used ,
            const vector<adten_t>& domain   ,
            const vector<adten_t>& rng_der ) const override {
            //
            // dom_der
            vector<adten_t> dom_der;
            //
            // dom_der
            if( rng_der[0].numel() == 0 ) {
                dom_der.push_back( adten_t( torch::empty( {0} ) ) );
            } else {
                vector<adten_t> domain_z;
                domain_z.push_back( domain[0] );
                domain_z.push_back( rng_der[0] );
                //
                dom_der = ad_tensor::call_atom(m_atom_id_z, domain_z);
            }
            if( get_trace() ) {
                cout << "AD reverse_der_y: domain =\n" << to_string(domain);
                cout << "AD reverse_der_y: dom_der =\n" << to_string(dom_der);
                cout << "AD reverse_der_y: rng_der =\n" << to_string(rng_der);
            }
            // opt
            std::optional< vector<adten_t> > opt = dom_der;
            return opt;
        }
    };
    // ----------------------------------------------------------------------
    // z(x, u) = 3 * x * x * u
    // ----------------------------------------------------------------------
    //
    // derive_atom_z_t
    class derive_atom_z_t : public ad_tensor::base_atom_t {
    public:
        // ctor
        derive_atom_z_t(void) {
            set_name("z");
        }
        // depend
        std::optional<ad_tensor::sparsity_t> depend(
            size_t                call_info) const override {
            ad_tensor::sparsity_t sparsity;
            sparsity.push_back( {0, 0} );
            sparsity.push_back( {0, 1} );
            //
            std::optional<ad_tensor::sparsity_t> opt = sparsity;
            return opt;
        }
        // forward
        std::optional< vector<Tensor> > forward(
            size_t                call_info ,
            const vector<bool>&   rng_used ,
            const vector<Tensor>& domain ) const override {
            //
            Tensor x  = domain[0];
            Tensor u  = domain[1];
            Tensor z  = 3.0 * x * x * u;
            // range
            vector<Tensor> range;
            range.push_back( z );
            if( get_trace() ) {
                cout << "forward_z: domain =\n" << to_string(domain);
                cout << "forward_z: range =\n" << to_string(range);
            }
            std::optional< vector<Tensor> > opt = range;
            return opt;
        }
        // forward_der
        std::optional< vector<Tensor> > forward_der(
            size_t                call_info,
            const vector<bool>&   rng_used,
            const vector<Tensor>& domain,
            const vector<Tensor>& dom_der) const override {
            //
            // dz
            Tensor x    = domain[0];
            Tensor u    = domain[1];
            Tensor dx   = dom_der[0];
            Tensor du   = dom_der[1];
            Tensor dz   = 6.0 * x * u * dx + 3.0 * x * x * du;
            //
            // rng_der
            vector<Tensor> rng_der;
            rng_der.push_back( dz );
            if( get_trace() ) {
                cout << "forward_der_z: domain =\n" << to_string(domain);
                cout << "forward_der_z: dom_der =\n" << to_string(dom_der);
                cout << "forward_der_z: rng_der =\n" << to_string(rng_der);
            }
            std::optional< vector<Tensor> > opt = rng_der;
            return opt;
        }
    };
}
// END_CPP
