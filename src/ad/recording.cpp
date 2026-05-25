// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/ad.hpp>
#include <ad_tensor/dev/tape.hpp>
#include <ad_tensor/dev/op_enum.hpp>
#include <ad_tensor/dev/agraph.hpp>
#include <ad_tensor/dev/user_assert.hpp>
//
namespace ad_tensor { // BEGIN_NAMESPACE_AD_TENSOR
/*
------------------------------------------------------------------------------
{xrst_begin start_recording usr}
{xrst_spell
    adom
}

Start Recording ad_t Operations
###############################
{xrst_literal ,
    include/ad_tensor/ad.hpp
    BEGIN_START_RECORDING, END_START_RECORDING
}

Recording
*********
This thread must not have a recording in progress
when ``start_recording`` is called.
The recording started by this call is stopped by calling
:ref:`stop_recording-name` .

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

{xrst_end start_recording}
*/
std::tuple< std::vector<ad_t>, std::vector<ad_t> > ad_t::start_recording(
        std::vector<at::Tensor>&& dom_par ,
        std::vector<at::Tensor>&& dom_var
)
{   //
    // tape
    dev::tape_t& tape = dev::this_threads_tape();
    //
    // next_tape_id
    // Since c++11, initialization of local static variables is thread safe.
    static size_t next_tape_id = 1;
    //
    // tape_id_mutex
    static std::mutex tape_id_mutex;
    //
    dev::user_assert( ! tape.recording() ,
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
    tape.m_tape_id   = tape_id;
    tape.m_recording = true;
    //
    // adom_par
    // tape.m_par: m_n_dom, m_op_seq
    ad_type_t parameter = ad_type_t::parameter;
    tape.m_par.m_n_dom    = dom_par.size();
    std::vector<ad_t> adom_par;
    {   auto itr_begin = std::make_move_iterator( dom_par.begin() );
        auto itr_end   = std::make_move_iterator( dom_par.end() );
        size_t index   = 0;
        for(auto itr = itr_begin; itr != itr_end; ++itr )
        {   tape.m_par.m_op_seq.push_back( dev::op_enum_t::dom );
            tape.m_par.m_arg_start.push_back( 0 );
            adom_par.push_back( ad_t( tape_id, index, *itr, parameter) );
            ++index;
        }
        assert( index == tape.m_par.m_n_dom );
    }
    //
    // adom_var
    // tape.m_var: m_n_dom, m_op_seq
    ad_type_t variable = ad_type_t::variable;
    tape.m_var.m_n_dom    = dom_var.size();
    std::vector<ad_t> adom_var;
    {   auto itr_begin = std::make_move_iterator( dom_var.begin() );
        auto itr_end   = std::make_move_iterator( dom_var.end() );
        size_t index   = 0;
        for(auto itr = itr_begin; itr != itr_end; ++itr )
        {   tape.m_var.m_op_seq.push_back( dev::op_enum_t::dom );
            tape.m_var.m_arg_start.push_back( 0 );
            adom_var.push_back( ad_t( tape_id, index, *itr, parameter) );
            ++index;
        }
        assert( index == tape.m_var.m_n_dom );
    }
    return std::tuple< std::vector<ad_t>, std::vector<ad_t> > (
        adom_par, adom_var
    );
}
/*
------------------------------------------------------------------------------
{xrst_begin stop_recording usr}
{xrst_spell
    adfn
    arange
}

Stop Recording ad_t Operations
##############################
{xrst_literal,
    include/ad_tensor/ad.hpp
    BEGIN_STOP_RECORDING, END_STOP_RECORDING
}

Recording
*********
This thread must have a recording in progress
when ``stop_recording`` is called.
Recording are started by calling :ref:`start_recording-name` .

arange
******
This vector of AD tensors specifies the range for the AD function *adfn* .

adfn
****
The operation sequence that was recording is transferred to this AD function.
The domain parameter and variable tensors for this function are defined
in the corresponding :ref:`start_recording-name` .

Example
*******
{xrst_literal ,
    examples/recording.cpp
    BEGIN_CPP, END_CPP
}

{xrst_end stop_recording}
*/
adfn_t ad_t::stop_recording(const std::vector<ad_t>& arange)
{   //
    // tape
    dev::tape_t& tape = dev::this_threads_tape();
    //
    dev::user_assert( tape.recording() ,
        "stop_recording: this threads tape is not recording"
    );
    assert( ! tape.is_empty() &&
        "stop_recording: a tape that is recording should not be empty"
    );
    assert( tape.m_par.m_arg_value.size() == tape.m_par.m_arg_type.size() &&
        "stop_recording: tape.m_par size of arg_value and arg_type not equal"
    );
    assert( tape.m_var.m_arg_value.size() == tape.m_var.m_arg_type.size() &&
        "stop_recording: tape.m_var size of arg_value and arg_type not equal"
    );
    //
    // tape
    tape.m_recording = false;
    tape.m_par.m_arg_start.push_back( tape.m_par.m_arg_value.size() );
    tape.m_var.m_arg_start.push_back( tape.m_var.m_arg_value.size() );
    //
    // adfn
    adfn_t adfn;
    //
    // adfn, tape
    adfn.m_con.swap( tape.m_con );
    adfn.m_par.swap( tape.m_par );
    adfn.m_var.swap( tape.m_var );
    //
    // adfn
    for(size_t i = 0; i < arange.size(); ++i)
    {   adfn.m_rng_index.push_back( arange.at(i).m_index );
        adfn.m_rng_ad_type.push_back( arange.at(i).m_ad_type );
    }
    //
    return adfn;
}
// ---------------------------------------------------------------------------
} // END_NAMESPACE_AD_TENSOR
