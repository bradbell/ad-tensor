#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <torch/torch.h>
//
#include <ad_tensor/devel/ad_type.hpp>
#include <ad_tensor/ad.hpp>
//
// BEGIN_START_RECORDING
namespace ad_tensor { 
std::tuple<ad_t, ad_t> start_recording (
    torch::Tensor&& dom_par, torch::Tensor&& dom_var 
) 
// END_START_RECORDING
{   //   
    /*
    user_assert(
        ! this_threads_tape.recording ,
        "start_recording: this_threads_tape is already recording\n"
    }
    devel_assert( this_threads_tape.is_empty() );
    //
    // tape_id, next_tape_id_global
    size_t tape_id;
    {   next_tape_id_mutex.lock();
        tape_id = next_tape_id_global;
        ++next_tape_id_global;
    }
    //
    // this_threads_tape: tape_id, recording, par.op_vec, var.op_vec
    this_threads_tape.tape_id   = tape_id;
    this_threads_tape.recording = true;
    this_threads_tape.par.op_vec.push( op_enum_t::dom );
    this_threads_tape.par.op_vec.push( op_enum_t::dom );
    */
    size_t tape_id = 0;
    //
    return {
        ad_t(tape_id, devel::ad_type_t::parameter, std::move(dom_par) ) ,
        ad_t(tape_id, devel::ad_type_t::variable,  std::move(dom_var) ) 
    };
} }
