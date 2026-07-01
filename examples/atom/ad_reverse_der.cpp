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
#include <ad_tensor/base_atom.hpp>
namespace {
    //
    // using
    using std::optional;
    using ad_tensor::vector;
    using ad_tensor::adten_t;
    using ad_tensor::options_t;
    using ad_tensor::base_atom_t;
    using at::Tensor;
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
            size_t                call_info ) const override {
            ad_tensor::sparsity_t sparsity;
            std::array<size_t, 2> pair = {0, 0};
            sparsity.push_back( pair );
            //
            std::optional<ad_tensor::sparsity_t> opt = sparsity;
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
            size_t                call_info ) const override{
            ad_tensor::sparsity_t sparsity;
            sparsity.push_back( {0, 0} );
            sparsity.push_back( {0, 1} );
            //
            std::optional<ad_tensor::sparsity_t> opt = sparsity;
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
        const options_t&      options   ,
        size_t                call_info ) {
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
        std::optional< vector<Tensor> > opt = rng_der;
        return opt;
    }
    //
    // ad_reverse_der_y
    std::optional< vector<adten_t> > ad_reverse_der_y(
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
        //
        std::optional< vector<adten_t> > opt = dom_der;
        return opt;
    }
    // ----------------------------------------------------------------------
    // z(x, dy) = 3 * x * x * dy
    // ----------------------------------------------------------------------
    // depend_z
    std::optional<ad_tensor::sparsity_t> depend_z(
        const options_t&      options   ,
        size_t                call_info ) {
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
        Tensor dy = domain[1];
        Tensor z  = 3.0 * x * x * dy;
        // range
        vector<Tensor> range;
        range.push_back( z );
        //
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
        Tensor dy   = domain[1];
        Tensor d_x  = dom_der[0];
        Tensor d_dy = dom_der[1];
        Tensor dz  = 6.0 * x * dy * d_x + 3.0 * x * x * d_dy;
        //
        // rng_der
        vector<Tensor> rng_der;
        rng_der.push_back( dz );
        //
        std::optional< vector<Tensor> > opt = rng_der;
        return opt;
    }
}
TEST(examples_atom, ad_reverse_der)  {
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
    atom_callback_y.set_ad_reverse_der(ad_reverse_der_y);
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
        arng_der, avar_all, apar_all
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
    vector<Tensor> rng_der = g.forward_der(dom_der, var_all);
    Tensor ddsum           = rng_der[0];
    //
    // check
    equal = ddsum.equal( 6.0 * x );
    EXPECT_TRUE(equal);
}
// END_CPP
