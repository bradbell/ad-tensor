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
namespace {
    //
    // using
    using ad_tensor::vector;
    using ad_tensor::adten_t;
    using ad_tensor::options_t;
    //
    using at::Tensor;
    using std::cout;
    //
    // atom_id_z
    size_t atom_id_z;
    //
    // ----------------------------------------------------------------------
    // y(x) = x * x * x
    // ----------------------------------------------------------------------
    // depend_y
    ad_tensor::sparsity_t depend_y(
        const options_t& options,
        size_t           call_info) {
        ad_tensor::sparsity_t sparsity;
        std::array<size_t, 2> pair = {0, 0};
        sparsity.push_back( pair );
        return sparsity;
    }
    //
    // forward
    vector<Tensor> forward_y(
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
        return range;
    }
    //
    // forward_der_y
    vector<Tensor> forward_der_y(
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
        return rng_der;
    }
    //
    // ad_forward_der_y
    vector<adten_t> ad_forward_der_y(
        const options_t&      options   ,
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
        return rng_der;
    }
    // ----------------------------------------------------------------------
    // z(x, dx) = 3 * x * x * dx
    // ----------------------------------------------------------------------
    // depend_z
    ad_tensor::sparsity_t depend_z(
        const options_t&      options  ,
        size_t                call_info) {
        ad_tensor::sparsity_t sparsity;
        sparsity.push_back( {0, 0} );
        sparsity.push_back( {0, 1} );
        return sparsity;
    }
    //
    // forward_z
    vector<Tensor> forward_z(
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
        return range;
    }
    //
    // forward_der_z
    vector<Tensor> forward_der_z(
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
        return rng_der;
    }
}
TEST(examples_atom, ad_forward_der)  {
    //
    // options
    ad_tensor::options_t options;
    //
    // atom_global
    ad_tensor::atom_global_t& atom_global =
        ad_tensor::atom_global_t::singleton();
    //
    // atom_callback_z
    ad_tensor::atom_callback_t atom_callback_z;
    atom_callback_z.set_name("z");
    atom_callback_z.set_options(options);
    atom_callback_z.set_depend(depend_z);
    atom_callback_z.set_forward(forward_z);
    atom_callback_z.set_forward_der(forward_der_z);
    //
    // atom_id_z
    atom_id_z = atom_global.store( atom_callback_z );
    //
    // atom_callback_y
    ad_tensor::atom_callback_t atom_callback_y;
    atom_callback_y.set_name("y");
    atom_callback_y.set_options(options);
    atom_callback_y.set_depend(depend_y);
    atom_callback_y.set_forward(forward_y);
    atom_callback_y.set_forward_der(forward_der_y);
    atom_callback_y.set_ad_forward_der(ad_forward_der_y);
    //
    // atom_id_y
    size_t atom_id_y = atom_global.store( atom_callback_y );
    //
    // x
    Tensor x = torch::tensor( {2.0, 3.0} );
    //
    // domain
    vector<Tensor> domain;
    domain.push_back(x);
    //
    // adomain
    vector<Tensor> par;
    auto [apar, adomain] = adten_t::start_recording(par, domain);
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
    std::tie(apar, adomain) = adten_t::start_recording(par, domain);
    //
    vector<adten_t> apar_all;
    vector<adten_t> avar_all = f.forward_var(apar_all, adomain, options);
    //
    // arng_der
    vector<adten_t> adom_der;
    adom_der.push_back( adten_t( torch::tensor( {1.0, 1.0} ) ) );
    vector<adten_t> arng_der = f.forward_der(
        apar_all, avar_all, adom_der, options
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
    vector<Tensor> var_all = g.forward_var(par_all, domain, options);
    vector<Tensor> range   = g.get_range(par_all, var_all);
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
    vector<Tensor> rng_der = g.forward_der(par_all, var_all, dom_der, options);
    Tensor ddsum           = rng_der[0];
    //
    // check
    EXPECT_EQ(ddsum.item<float>(), (6.0 * x).sum().item<float>() );
}
// END_CPP
