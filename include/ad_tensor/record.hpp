#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <torch/torch.h>
//
#include <ad_tensor/devel/ad_type.hpp>
#include <ad_tensor/devel/op_enum.hpp>
#include <ad_tensor/devel/tape.hpp>
#include <ad_tensor/ad.hpp>
//
// BEGIN_START_RECORDING
namespace ad_tensor { 
std::tuple<ad_t, ad_t> start_recording (
    torch::Tensor&& dom_par, torch::Tensor&& dom_var 
) 
// END_START_RECORDING
{   //   
    //
    // next_tape_id
    // Since c++11, initialization of local static variables is thread safe.
    static size_t next_tape_id = 1;
    //
    // tape_id_mutex
    static std::mutex tape_id_mutex;
    //
    assert( ! devel::this_threads_tape.recording &&
        "start_recording: this threads tape is already recording"
    );
    assert( devel::this_threads_tape.is_empty() &&
        "start_recording: a tape that is not recording should be empty"
    );
    //
    // tape_id, next_tape_id
    size_t tape_id;
    {   std::lock_guard<std::mutex> lock(tape_id_mutex);
        tape_id = next_tape_id;
        ++next_tape_id;
    }
    //
    // this_threads_tape: tape_id, recording, par.op_vec, var.op_vec
    devel::this_threads_tape.tape_id   = tape_id;
    devel::this_threads_tape.recording = true;
    devel::this_threads_tape.par.op_vec.push_back( devel::op_enum_t::dom );
    devel::this_threads_tape.par.op_vec.push_back( devel::op_enum_t::dom );
    //
    return {
        ad_t(tape_id, devel::ad_type_t::parameter, std::move(dom_par) ) ,
        ad_t(tape_id, devel::ad_type_t::variable,  std::move(dom_var) ) 
    };
} }
