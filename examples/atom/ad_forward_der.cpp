// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <ad_tensor/adfn.hpp>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/options.hpp>
#include <ad_tensor/atom.hpp>
#include <ad_tensor/sparsity.hpp>
#include <ad_tensor/to_string.hpp>
#include <ad_tensor/base_atom.hpp>
namespace {
    //
    // using
    using std::optional;
    using ad_tensor::vector;
    using ad_tensor::adten_t;
    using ad_tensor::options_t;
    using ad_tensor::base_atom_t;
    //
    using at::Tensor;
    using std::cout;
    //
    // atom_id_z
    size_t atom_id_z;
    //
    // derive_atom_y
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
            std::array<size_t, 2> pair = {0, 0};
            sparsity.push_back( pair );
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
    };
    //
    // derive_atom_z
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
            Tensor dx = domain[1];
            Tensor z  = 3.0 * x * x * dx;
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
            Tensor dx   = domain[1];
            Tensor d_x  = dom_der[0];
            Tensor d_dx = dom_der[1];
            Tensor dz   = 6.0 * x * dx * d_x + 3.0 * x * x * d_dx;
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
    //
    // base_atom_y_ptr, base_atom_z
    std::unique_ptr<base_atom_t> base_atom_y_ptr =
        std::make_unique<derive_atom_y_t>();
    std::unique_ptr<base_atom_t> base_atom_z_ptr =
        std::make_unique<derive_atom_z_t>();
    //
    // ----------------------------------------------------------------------
    // y(x) = x * x * x
    // ----------------------------------------------------------------------
    // depend_y
    std::optional<ad_tensor::sparsity_t> depend_y(
        const options_t& options,
        size_t           call_info) {
        ad_tensor::sparsity_t sparsity;
        std::array<size_t, 2> pair = {0, 0};
        sparsity.push_back( pair );
        //
        std::optional<ad_tensor::sparsity_t> opt = sparsity;
        return opt;
    }
    //
    // forward
    std::optional< vector<Tensor> > forward_y(
        const options_t&      options   ,
        size_t                call_info ,
        const vector<bool>&   rng_used ,
        const vector<Tensor>& domain ) {
        //
        Tensor x = domain[0];
        //
        // range
        vector<Tensor> range;
        range.push_back( x * x * x );
        //
        if( options.get_trace() ) {
            cout << "forward_y: domain =\n" << to_string(domain);
            cout << "forward_y: range =\n" << to_string(range);
        }
        std::optional< vector<Tensor> > opt = range;
        return opt;
    }
    //
    // forward_der_y
    std::optional< vector<Tensor> > forward_der_y(
        const options_t&      options   ,
        size_t                call_info ,
        const vector<bool>&   rng_used  ,
        const vector<Tensor>& domain    ,
        const vector<Tensor>& dom_der   ) {
        //
        Tensor x  = domain[0];
        Tensor dx = dom_der[0];
        //
        // range
        vector<Tensor> rng_der;
        rng_der.push_back( 3.0 * x * x * dx );
        //
        if( options.get_trace() ) {
            cout << "forward_der_y: domain =\n" << to_string(domain);
            cout << "forward_der_y: dom_der =\n" << to_string(dom_der);
            cout << "forward_der_y: rng_der =\n" << to_string(rng_der);
        }
        std::optional< vector<Tensor> > opt = rng_der;
        return opt;
    }
    //
    // ad_forward_der_y
    std::optional< vector<adten_t> > ad_forward_der_y(
        const options_t&       options   ,
        size_t                 call_info ,
        const vector<bool>&    rng_used ,
        const vector<adten_t>& domain   ,
        const vector<adten_t>& dom_der ) {
        //
        vector<adten_t> domain_z;
        domain_z.push_back( domain[0] );
        domain_z.push_back( dom_der[0] );
        //
        // rng_der
        vector<adten_t> rng_der = adten_t::call_atom(
            atom_id_z, call_info, domain_z
        );
        if( options.get_trace() ) {
            cout << "ad_forward_der_y: domain =\n" << to_string(domain);
            cout << "ad_forward_der_y: dom_der =\n" << to_string(dom_der);
            cout << "ad_forward_der_y: rng_der =\n" << to_string(rng_der);
        }
        //
        std::optional< vector<adten_t> > opt = rng_der;
        return opt;
    }
    // ----------------------------------------------------------------------
    // z(x, dx) = 3 * x * x * dx
    // ----------------------------------------------------------------------
    // depend_z
    std::optional<ad_tensor::sparsity_t> depend_z(
        const options_t&      options  ,
        size_t                call_info) {
        ad_tensor::sparsity_t sparsity;
        sparsity.push_back( {0, 0} );
        sparsity.push_back( {0, 1} );
        //
        std::optional<ad_tensor::sparsity_t> opt = sparsity;
        return opt;
    }
    //
    // forward_z
    std::optional< vector<Tensor> > forward_z(
        const options_t&      options   ,
        size_t                call_info ,
        const vector<bool>&   rng_used ,
        const vector<Tensor>& domain ) {
        //
        Tensor x  = domain[0];
        Tensor dx = domain[1];
        Tensor z  = 3.0 * x * x * dx;
        // range
        vector<Tensor> range;
        range.push_back( z );
        if( options.get_trace() ) {
            cout << "forward_z: domain =\n" << to_string(domain);
            cout << "forward_z: range =\n" << to_string(range);
        }
        std::optional< vector<Tensor> > opt = range;
        return opt;
    }
    //
    // forward_der_z
    std::optional< vector<Tensor> > forward_der_z(
        const options_t&      options   ,
        size_t                call_info,
        const vector<bool>&   rng_used,
        const vector<Tensor>& domain,
        const vector<Tensor>& dom_der) {
        //
        // dz
        Tensor x    = domain[0];
        Tensor dx   = domain[1];
        Tensor d_x  = dom_der[0];
        Tensor d_dx = dom_der[1];
        Tensor dz   = 6.0 * x * dx * d_x + 3.0 * x * x * d_dx;
        //
        // rng_der
        vector<Tensor> rng_der;
        rng_der.push_back( dz );
        if( options.get_trace() ) {
            cout << "forward_der_z: domain =\n" << to_string(domain);
            cout << "forward_der_z: dom_der =\n" << to_string(dom_der);
            cout << "forward_der_z: rng_der =\n" << to_string(rng_der);
        }
        std::optional< vector<Tensor> > opt = rng_der;
        return opt;
    }
}
TEST(examples_atom, ad_forward_der)  {
    //
    // atom_global
    ad_tensor::atom_global_t& atom_global =
        ad_tensor::atom_global_t::singleton();
    //
    // atom_callback_z
    ad_tensor::atom_callback_t atom_callback_z;
    atom_callback_z.set_name("z");
    atom_callback_z.set_depend(depend_z);
    atom_callback_z.set_forward(forward_z);
    atom_callback_z.set_forward_der(forward_der_z);
    //
    // atom_id_z
    atom_id_z = atom_global.store(atom_callback_z, base_atom_z_ptr);
    //
    // atom_callback_y
    ad_tensor::atom_callback_t atom_callback_y;
    atom_callback_y.set_name("y");
    atom_callback_y.set_depend(depend_y);
    atom_callback_y.set_forward(forward_y);
    atom_callback_y.set_forward_der(forward_der_y);
    atom_callback_y.set_ad_forward_der(ad_forward_der_y);
    //
    // atom_id_y
    size_t atom_id_y = atom_global.store(atom_callback_y, base_atom_y_ptr);
    //
    // x
    Tensor x = torch::tensor( {2.0, 3.0} );
    //
    // domain
    vector<Tensor> domain;
    domain.push_back(x);
    //
    // adomain
    vector<adten_t> adomain = adten_t::start_recording(domain);
    //
    // ay
    size_t call_info = 0;
    vector<adten_t> ay = adten_t::call_atom(atom_id_y, call_info, adomain);
    //
    adten_t asum;
    asum = ay[0].sum();
    //
    // arange
    vector<adten_t> arange;
    arange.push_back( asum );
    //
    // range = f(domain) = (x * x * x).sum()
    ad_tensor::adfn_t f = adten_t::stop_recording(arange, "f");
    //
    // adomain
    adomain = adten_t::start_recording(domain);
    //
    vector<adten_t> apar_all;
    vector<adten_t> avar_all = f.forward_var(adomain);
    //
    // arng_der
    vector<adten_t> adom_der;
    adom_der.push_back( adten_t( torch::tensor( {1.0, 1.0} ) ) );
    vector<adten_t> arng_der = f.forward_der(
        adom_der, avar_all, apar_all
    );
    //
    // arng_der = g(domain) = f'(domain) = (3 * x * x).sum()
    ad_tensor::adfn_t g = adten_t::stop_recording(arng_der, "g");
    //
    // x, domain
    x = torch::tensor( {3.0, 4.0} );
    domain[0] = x;
    //
    // dsum
    vector<Tensor> par_all;
    vector<Tensor> var_all = g.forward_var(domain);
    vector<Tensor> range   = g.get_range(var_all);
    Tensor         dsum    = range[0];
    //
    // check
    EXPECT_EQ(dsum.item<float>(), (3.0 * x * x).sum().item<float>() );
    //
    // dx, dom_der
    Tensor dx = torch::tensor( {1.0, 1.0} );
    vector<Tensor> dom_der;
    dom_der.push_back( dx );
    //
    // ddsum
    vector<Tensor> rng_der = g.forward_der(dom_der, var_all);
    Tensor ddsum           = rng_der[0];
    //
    // check
    EXPECT_EQ(ddsum.item<float>(), (6.0 * x).sum().item<float>() );
}
// END_CPP
