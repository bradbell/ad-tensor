#pragma once
// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin record_start usr}
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
This thread must not have a recording in progress
when ``record_t::start`` is called.
The recording started by this call is stopped by calling
:ref:`record_t::stop <record_stop-name>` .

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
This is the domain parameter AD tensor. Dependencies with respect to this
object will be recorded.

adom_var
********
This is the domain variable AD tensor. The recording can be used to compute
derivatives with respect to this object.

Example
*******
{xrst_literal ,
    examples/record.cpp
}

{xrst_end record_start}
-------------------------------------------------------------------------------
{xrst_begin record_stop usr}
{xrst_spell
    adfn
    arange
}

Stop Recording ad_t Operations
##############################
{xrst_literal,
    BEGIN_STOP, END_STOP
}

Recording
*********
This thread must have a recording in progress
when ``record_t::stop`` is called.
The recording started by calling
:ref:`record_t::start <record_start-name>` .

arange
******
This AD tensor specifies the range tensor for the AD function *adfn* .

adfn
****
The operation sequence that was recording is transferred to this AD function.
The domain parameter and variable tensors for this function are defined
in the corresponding :ref:`record_t::start<record_start-name>` .

Example
*******
{xrst_literal ,
    examples/record.cpp
}

{xrst_end record_stop}
*/
#include <torch/torch.h>
//
#include <ad_tensor/ad.hpp>
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/ad_type.hpp>
#include <ad_tensor/devel/op_enum.hpp>
#include <ad_tensor/devel/tape.hpp>
//
namespace ad_tensor { class record_t {
private:
    typedef devel::op_enum_t op_enum_t;
    typedef devel::tape_t    tape_t;
public:
    // BEGIN_START
    // auto [adom_par, adom_var] = ad_tensor::record_t::start(dom_par, dom_var)
    static std::tuple<ad_t, ad_t> start(
        at::Tensor&& dom_par, at::Tensor&& dom_var
    )
    // END_START
    {   //
        // tape
        tape_t& tape = devel::this_threads_tape;
        //
        // next_tape_id
        // Since c++11, initialization of local static variables is thread safe.
        static size_t next_tape_id = 1;
        //
        // tape_id_mutex
        static std::mutex tape_id_mutex;
        //
        assert( ! tape.recording() &&
            "record_start: this threads tape is already recording"
        );
        assert( tape.is_empty() &&
            "record_start: a tape that is not recording should be empty"
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
        // use nop for op_index equal to zero (corresponds to domain values)
        tape.tape_id_   = tape_id;
        tape.recording_ = true;
        tape.par_.op_vec.push_back( op_enum_t::nop );
        tape.var_.op_vec.push_back( op_enum_t::nop );
        //
        size_t index = 0;
        return {
            ad_t(tape_id, ad_type_t::parameter, index, std::move(dom_par) ) ,
            ad_t(tape_id, ad_type_t::variable,  index, std::move(dom_var) )
        };
    }
    // BEGIN_STOP
    // adfn = ad_tensor::record_t::stop(arnage)
    static adfn_t stop(const ad_t& arange)
    // END_STOP
    {   //
        // tape
        tape_t& tape = devel::this_threads_tape;
        //
        assert( tape.recording() &&
            "record::stop: this threads tape is not recording"
        );
        assert( ! tape.is_empty() &&
            "record_stop: a tape that is recording should not be empty"
        );
        //
        // tape
        tape.recording_ = false;
        //
        // adfn
        adfn_t adfn;
        //
        // adfn, tape
        adfn.con_.swap( tape.con_ );
        adfn.par_.swap( tape.par_ );
        adfn.var_.swap( tape.var_ );
        //
        // adfn
        adfn.rng_index_   = arange.index_;
        adfn.rng_ad_type_ = arange.ad_type_;
        //
        return adfn;
    }
}; }
