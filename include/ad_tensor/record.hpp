#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin start_recording usr}
{xrst_spell
    adom
}

Start Recording ad_t Operations
###############################
{xrst_literal ,
    BEGIN_START, END_START
}

Recording
*********
This thread must not currently have a recording in progress
when start_recording is called.
The recording started by this call is stopped by calling stop_recording.

dom_par
*******
This is the domain parameter tensor. Dependencies with respect to this
tensor will be recorded.

dom_var
*******
This is the domain variable tensor. The recording can be used to compute
derivatives with respect to this tensor.

adom_par
********
This is the domain parameter AD object. Dependencies with respect to this
object will be recorded.

adom_var
********
This is the domain variable AD object. The recording can be used to compute
derivatives with respect to this object.

Example
*******
{xrst_literal ,
    examples/start_recording.cpp
}

{xrst_end start_recording}
*/
#include <torch/torch.h>
//
#include <ad_tensor/devel/ad_type.hpp>
#include <ad_tensor/devel/op_enum.hpp>
#include <ad_tensor/devel/tape.hpp>
#include <ad_tensor/ad.hpp>
//
namespace ad_tensor { class record_t {
public:
    // BEGIN_START
    // auto [adom_par, adom_var] = ad_tensor::record::start(dom_par, dom_var)
    static std::tuple<ad_t, ad_t> start(
        torch::Tensor&& dom_par, torch::Tensor&& dom_var
    )
    // END_START
    {   //
        // next_tape_id
        // Since c++11, initialization of local static variables is thread safe.
        static size_t next_tape_id = 1;
        //
        // tape_id_mutex
        static std::mutex tape_id_mutex;
        //
        assert( ! devel::this_threads_tape.recording() &&
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
        devel::this_threads_tape.tape_id_   = tape_id;
        devel::this_threads_tape.recording_ = true;
        devel::this_threads_tape.par_.op_vec.push_back( devel::op_enum_t::dom );
        devel::this_threads_tape.var_.op_vec.push_back( devel::op_enum_t::dom );
        //
        return {
            ad_t(tape_id, devel::ad_type_t::parameter, std::move(dom_par) ) ,
            ad_t(tape_id, devel::ad_type_t::variable,  std::move(dom_var) )
        };
    }
}; }
