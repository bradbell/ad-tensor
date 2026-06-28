// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
// BEGIN_CPP
#include <ad_tensor/adfn.hpp>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ad_tensor/adten.hpp>
#include <ad_tensor/atom.hpp>
#include <ad_tensor/sparsity.hpp>
#include <ad_tensor/options.hpp>
namespace {
    //
    // using
    using ad_tensor::vector;
    using ad_tensor::adten_t;
    using ad_tensor::options_t;
    using at::Tensor;
    //
    // atom_id_z
    size_t atom_id_z;
    //
    // ----------------------------------------------------------------------
    // y(x) = x * x * x
    // ----------------------------------------------------------------------
    // depend_y
    ad_tensor::sparsity_t depend_y(
        const options_t&      options   ,
        size_t                call_info ) {
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
        vector<Tensor> range;
        range.push_back( 3.0 * x * x * dx );
        return range;
    }
    //
    // ad_reverse_der_y
    vector<adten_t> ad_reverse_der_y(
        const options_t&       options   ,
        size_t                 call_info ,
        const vector<bool>&    rng_used ,
        const vector<adten_t>& domain   ,
        const vector<adten_t>& rng_der ) {
        //
        if( rng_der[0].numel() == 0 ) {
            return rng_der;
        }
        //
        vector<adten_t> domain_z;
        domain_z.push_back( domain[0] );
        domain_z.push_back( rng_der[0] );
        //
        // dom_der
        vector<adten_t> dom_der = adten_t::call_atom(
            atom_id_z, call_info, domain_z
        );
        return dom_der;
    }
    // ----------------------------------------------------------------------
    // z(x, dy) = 3 * x * x * dy
    // ----------------------------------------------------------------------
    // depend_z
    ad_tensor::sparsity_t depend_z(
        const options_t&      options   ,
        size_t                call_info ) {
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
        Tensor dy = domain[1];
        Tensor z  = 3.0 * x * x * dy;
        // range
        vector<Tensor> range;
        range.push_back( z );
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
        Tensor dy   = domain[1];
        Tensor d_x  = dom_der[0];
        Tensor d_dy = dom_der[1];
        Tensor dz  = 6.0 * x * dy * d_x + 3.0 * x * x * d_dy;
        //
        // rng_der
        vector<Tensor> rng_der;
        rng_der.push_back( dz );
        return rng_der;
    }
}
TEST(examples_atom, ad_reverse_der)  {
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
    atom_callback_z.set_options(options);
    atom_callback_y.set_depend(depend_y);
    atom_callback_y.set_forward(forward_y);
    atom_callback_y.set_forward_der(forward_der_y);
    atom_callback_y.set_ad_reverse_der(ad_reverse_der_y);
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
    // range = f(domain)
    ad_tensor::adfn_t f = adten_t::stop_recording(arange, "f");
    //
    // adomain
    adomain = adten_t::start_recording(domain);
    //
    vector<adten_t> apar_all;
    vector<adten_t> avar_all = f.forward_var(adomain);
    //
    // adom_der
    vector<adten_t> arng_der;
    arng_der.push_back( adten_t( torch::tensor( 1.0 ) ) );
    vector<adten_t> adom_der = f.reverse_der(
        apar_all, avar_all, arng_der, options
    );
    //
    // adom_der = g(domain) = f'(domain)
    ad_tensor::adfn_t g = adten_t::stop_recording(adom_der, "g");
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
    bool equal = dsum.equal(3.0 * x * x);
    EXPECT_TRUE(equal);
    //
    // dx, dom_der
    Tensor dx = torch::tensor( {1.0, 1.0} );
    vector<Tensor> dom_der;
    dom_der.push_back( dx );
    //
    // ddsum
    vector<Tensor> rng_der = g.forward_der(dom_der, var_all, par_all);
    Tensor ddsum           = rng_der[0];
    //
    // check
    equal = ddsum.equal( 6.0 * x );
    EXPECT_TRUE(equal);
}
// END_CPP
