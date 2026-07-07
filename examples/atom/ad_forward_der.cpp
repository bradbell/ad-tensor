// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin atom_ad_forward_der usr}
{xrst_spell
    dx
    adten
    nowrap
    lll
}


Recording Forward Derivatives That Use Atomic Functions
#######################################################



where sum above sums the elements of the corresponding tensor.


Derived Classes
***************
This example defines an atomic function for y and z.
The atomic function for y uses z when it is recording forward derivatives.

Tracing
*******
If :ref:`atom_non_virtual@trace` is true,
this derived class prints some information about its internal calculations.

derive_atom_y_t
***************
This derived atomic class implements the following function
and domain direction derivative:

.. math::

    y(x)        &= x^3 \\
    y'(x) * dx  &= 3 * x^2 * dx \\

In addition, it uses the z(x, dx) atomic function for the
domain direction derivative when the argument type is vector<adten_t>.

derive_atom_z_t
***************
This derived atomic class implements the following function
and domain direction derivative where u represents dx above:

.. math::

    z(x, u)                          &= 3 * x^2 * u  \\
    z_x(x, u) * dx + z_u(x, u) * du  &= 6 * x * u * dx + 3 * x^2 * du


AD Functions
************

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
namespace {
    //
    // using
    using std::optional;
    using ad_tensor::vector;
    using ad_tensor::adten_t;
    using ad_tensor::base_atom_t;
    //
    using at::Tensor;
    using std::cout;
    //
    // atom_id_z
    // declare this here so it can be used by the atomic function for y(x)
    size_t atom_id_z;
    //
    // ----------------------------------------------------------------------
    // y(x) = x * x * x
    // ----------------------------------------------------------------------
    //
    // derive_atom_y_t
    class derive_atom_y_t : public base_atom_t {
    public:
        // ctor
        derive_atom_y_t(void) {
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
        std::optional< vector<adten_t> > forward_der(
            size_t                 call_info ,
            const vector<bool>&    rng_used ,
            const vector<adten_t>& domain   ,
            const vector<adten_t>& dom_der ) const override {
            //
            vector<adten_t> domain_z;
            domain_z.push_back( domain[0] );
            domain_z.push_back( dom_der[0] );
            //
            // rng_der
            vector<adten_t> rng_der = ad_tensor::call_atom(
                atom_id_z, call_info, domain_z
            );
            if( get_trace() ) {
                cout << "AD forward_der_y: domain =\n" << to_string(domain);
                cout << "AD forward_der_y: dom_der =\n" << to_string(dom_der);
                cout << "AD forward_der_y: rng_der =\n" << to_string(rng_der);
            }
            //
            std::optional< vector<adten_t> > opt = rng_der;
            return opt;
        }
    };
    // ----------------------------------------------------------------------
    // z(x, u) = 3 * x * x * u
    // ----------------------------------------------------------------------
    //
    // derive_atom_z_t
    class derive_atom_z_t : public base_atom_t {
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
    // -----------------------------------------------------------------------
    //
    // base_atom_y_ptr, base_atom_z
    std::unique_ptr<base_atom_t> base_atom_y_ptr =
        std::make_unique<derive_atom_y_t>();
    std::unique_ptr<base_atom_t> base_atom_z_ptr =
        std::make_unique<derive_atom_z_t>();
}
TEST(examples_atom, ad_forward_der)  {
    //
    // atom_id_z
    atom_id_z = ad_tensor::make_atom(base_atom_z_ptr);
    //
    // atom_id_y
    size_t atom_id_y = ad_tensor::make_atom(base_atom_y_ptr);
    //
    // v, av
    vector<Tensor> v;
    v.push_back( torch::tensor( {0.0, 0.0} ) );
    vector<adten_t> av = adten_t::start_recording(v);
    //
    // ay
    size_t call_info = 0;
    vector<adten_t> ay = ad_tensor::call_atom(atom_id_y, call_info, av);
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
