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
//
namespace {
    //
    // vector
    using ad_tensor::vector;
    using ad_tensor::adten_t;
    using at::Tensor;
    //
    // depend
    ad_tensor::sparsity_t depend(size_t call_info) {
        ad_tensor::sparsity_t sparsity;
        std::array<size_t, 2> pair = {0, 0};
        sparsity.push_back( pair );
        return sparsity;
    }
    //
    // forward
    vector<at::Tensor> forward(
        size_t                    call_info ,
        const vector<bool>&       rng_used ,
        const vector<at::Tensor>& domain ) {
        //
        // range
        vector<at::Tensor> range;
        range.push_back( domain[0] * domain[0] );
        return range;
    }
    //
    // forward_der
    vector<at::Tensor> forward_der(
        size_t                    call_info ,
        const vector<bool>&       rng_used ,
        const vector<at::Tensor>& domain   ,
        const vector<at::Tensor>& dom_der ) {
        //
        // range
        vector<at::Tensor> range;
        range.push_back( 2.0 * domain[0] * dom_der[0] );
        return range;
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
    // y = f(x)
    ad_tensor::adfn_t f = adten_t::stop_recording(ay);
    //
    // x
    x[0] = torch:: tensor( {3.0, 4.0} );
    //
    // y
    vector<Tensor> par_all;
    vector<Tensor> var_all = f.forward_var(par_all, x, options);
    vector<Tensor> y       = f.get_range(par_all, var_all);
    //
    // check
    bool equal = y[0].equal( x[0] * x[0] );
    EXPECT_TRUE( equal );
    //
    // dx, dy
    vector<Tensor> dx;
    dx.push_back( torch::tensor( {4.0, 5.0} ) );
    vector<Tensor> dy = f.forward_der(par_all, var_all, dx, options);
    //
    // check
    equal = dy[0].equal( 2.0 * x[0] * dx[0] );
    EXPECT_TRUE( equal );
}
// END_CPP
