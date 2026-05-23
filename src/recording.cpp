// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <torch/torch.h>
//
#include <ad_tensor/ad.hpp>
#include <ad_tensor/adfn.hpp>
#include <ad_tensor/ad_type.hpp>
#include <ad_tensor/devel/op_enum.hpp>
#include <ad_tensor/devel/tape.hpp>
#include <ad_tensor/recording.hpp>
//
namespace ad_tensor { // BEGIN_NAMESPACE_AD_TENSOR
/*
------------------------------------------------------------------------------
{xrst_begin_parent recording usr}

The recording Class
###################
This class is special in two ways;
It only has static member functions and its name does not end with ``_t`` .

{xrst_end recording}
------------------------------------------------------------------------------
{xrst_begin recording_start usr}
{xrst_spell
    adom
}

Start Recording ad_t Operations
###############################
{xrst_literal ,
    include/ad_tensor/recording.hpp
    BEGIN_START, END_START
}

Recording
*********
This thread must not have a recording in progress
when ``recording::start`` is called.
The recording started by this call is stopped by calling
:ref:`recording::stop <recording_stop-name>` .

dom_par
*******
This is the domain parameter vector of tensors.

dom_var
*******
This is the domain variable vector of tensors.

adom_par
********
This is the domain parameter vector of AD tensors.
Dependencies with respect to these tensors will be recorded.

adom_var
********
This is the domain variable vectors of AD tensors.
The recording can be used to compute derivatives with respect to these tensors.

Example
*******
{xrst_literal ,
    examples/recording.cpp
    BEGIN_CPP, END_CPP
}

{xrst_end recording_start}
*/
std::tuple< std::vector<ad_t>, std::vector<ad_t> > recording::start(
        std::vector<at::Tensor>&& dom_par ,
        std::vector<at::Tensor>&& dom_var
)
{   //
    // tape
    devel::tape_t& tape = devel::this_threads_tape;
    //
    // next_tape_id
    // Since c++11, initialization of local static variables is thread safe.
    static size_t next_tape_id = 1;
    //
    // tape_id_mutex
    static std::mutex tape_id_mutex;
    //
    assert( ! tape.recording() &&
        "recording_start: this threads tape is already recording"
    );
    assert( tape.is_empty() &&
        "recording_start: a tape that is not recording should be empty"
    );
    //
    // tape_id, next_tape_id
    size_t tape_id;
    {   std::lock_guard<std::mutex> lock(tape_id_mutex);
        tape_id = next_tape_id;
        ++next_tape_id;
    }
    //
    // tape: tape_id, recording
    tape.tape_id_   = tape_id;
    tape.recording_ = true;
    //
    // adom_par
    // tape.par_: n_dom_, op_vec_
    ad_type_t parameter = ad_type_t::parameter;
    tape.par_.n_dom_     = dom_par.size();
    std::vector<ad_t> adom_par;
    for(size_t index = 0; index < dom_par.size(); ++index)
    {   tape.par_.op_vec_.push_back( devel::op_enum_t::dom );
        adom_par.push_back( ad_t(
            tape_id, parameter, index, dom_par[index].clone()
        ) );
    }
    //
    // adom_var
    // tape.var_: n_dom_, op_vec_
    ad_type_t variable = ad_type_t::variable;
    tape.var_.n_dom_    = dom_var.size();
    std::vector<ad_t> adom_var;
    for(size_t index = 0; index < dom_var.size(); ++index)
    {   tape.var_.op_vec_.push_back( devel::op_enum_t::dom );
        adom_var.push_back( ad_t(
            tape_id, variable, index, dom_var[index].clone()
        ) );
    }

    return std::tuple< std::vector<ad_t>, std::vector<ad_t> > (
        adom_par, adom_var
    );
}
/*
------------------------------------------------------------------------------
{xrst_begin recording_stop usr}
{xrst_spell
    adfn
    arange
}

Stop Recording ad_t Operations
##############################
{xrst_literal,
    include/ad_tensor/recording.hpp
    BEGIN_STOP, END_STOP
}

Recording
*********
This thread must have a recording in progress
when ``recording::stop`` is called.
The recording started by calling
:ref:`recording::start <recording_start-name>` .

arange
******
This vector of AD tensors specifies the range for the AD function *adfn* .

adfn
****
The operation sequence that was recording is transferred to this AD function.
The domain parameter and variable tensors for this function are defined
in the corresponding :ref:`recording::start<recording_start-name>` .

Example
*******
{xrst_literal ,
    examples/recording.cpp
    BEGIN_CPP, END_CPP
}

{xrst_end recording_stop}
*/
// stop
adfn_t recording::stop(const std::vector<ad_t>& arange)
{   //
    // tape
    devel::tape_t& tape = devel::this_threads_tape;
    //
    assert( tape.recording() &&
        "recording::stop: this threads tape is not recording"
    );
    assert( ! tape.is_empty() &&
        "recording_stop: a tape that is recording should not be empty"
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
    for(size_t i = 0; i < arange.size(); ++i)
    {   adfn.rng_index_.push_back( arange.at(i).index_ );
        adfn.rng_ad_type_.push_back( arange.at(i).ad_type_ );
    }
    //
    return adfn;
}
// ---------------------------------------------------------------------------
} // END_NAMESPACE_AD_TENSOR
