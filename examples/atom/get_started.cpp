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
//
namespace {
    //
    // vector
    using ad_tensor::vector;
    using ad_tensor::adten_t;
    using ad_tensor::options_t;
    using at::Tensor;
    //
    // depend
    ad_tensor::sparsity_t depend(
        const options_t&      options   ,
        size_t                call_info ) {
        ad_tensor::sparsity_t sparsity;
        sparsity.push_back( {0, 0} );
        return sparsity;
    }
    //
    // forward
    vector<Tensor> forward(
        const options_t&      options   ,
        size_t                call_info ,
        const vector<bool>&   rng_used ,
        const vector<Tensor>& domain ) {
        //
        // range
        vector<Tensor> range;
        range.push_back( domain[0] * domain[0] );
        return range;
    }
    //
    // forward_der
    vector<Tensor> forward_der(
        const options_t&      options   ,
        size_t                call_info ,
        const vector<bool>&   rng_used ,
        const vector<Tensor>& domain   ,
        const vector<Tensor>& dom_der ) {
        //
        // rng_der
        vector<Tensor> rng_der;
        rng_der.push_back( 2.0 * domain[0] * dom_der[0] );
        return rng_der;
    }
    //
    // reverse_der
    vector<Tensor> reverse_der(
        const options_t&      options   ,
        size_t                call_info ,
        const vector<bool>&   rng_used ,
        const vector<Tensor>& domain   ,
        const vector<Tensor>& rng_der ) {
        //
        // dom_der
        vector<Tensor> dom_der;
        if( rng_der[0].numel() == 0 ) {
            dom_der.push_back( torch::empty( {0} ) );
        } else {
            dom_der.push_back( 2.0 * domain[0] * rng_der[0] );
        }
        return dom_der;
    }
}
TEST(examples_atom, get_started)  {
    //
    // options
    ad_tensor::options_t options;
    //
    // atom_callback
    ad_tensor::atom_callback_t atom_callback;
    atom_callback.set_name("square");
    atom_callback.set_depend(depend);
    atom_callback.set_forward(forward);
    atom_callback.set_forward_der(forward_der);
    atom_callback.set_reverse_der(reverse_der);
    //
    // atom_id
    ad_tensor::atom_global_t& atom_global =
        ad_tensor::atom_global_t::singleton();
    size_t atom_id = atom_global.store( atom_callback );
    //
    // x
    // We use x for the adfn domain variables
    vector<Tensor> x;
    x.push_back( torch::tensor( {2.0, 3.0} ) );
    //
    // ax
    vector<Tensor> p;
    auto [ap, ax] = adten_t::start_recording(p, x);
    //
    // ay
    size_t call_info = 0;
    vector<adten_t> ay = adten_t::call_atom(atom_id, call_info, ax);
    //
    // az
    vector<adten_t> az;
    az.push_back( ay[0].sum() );
    //
    // z = f(x)
    ad_tensor::adfn_t f = adten_t::stop_recording(az);
    //
    // x
    x[0] = torch:: tensor( {3.0, 4.0} );
    //
    // z
    vector<Tensor> par_all;
    vector<Tensor> var_all = f.forward_var(par_all, x, options);
    vector<Tensor> z       = f.get_range(par_all, var_all);
    //
    // check
    EXPECT_EQ( z[0].item<float>(), (x[0] * x[0]).sum().item<float>() );
    //
    // dx, dz
    // forward mode derivative
    vector<Tensor> dx;
    dx.push_back( torch::tensor( {4.0, 5.0} ) );
    vector<Tensor> dz = f.forward_der(par_all, var_all, dx, options);
    //
    // check
    EXPECT_EQ( dz[0].item<float>(), (2.0 * x[0] * dx[0]).sum().item<float>() );
    //
    // dz, dx
    // reverse mode derivative
    dz[0] = torch::tensor( 6.0 );
    dx = f.reverse_der(par_all, var_all, dz, options);
    //
    // check
    bool equal;
    equal = dx[0].equal( dz[0] * 2.0 * x[0] );
    EXPECT_TRUE( equal );
}
// END_CPP
