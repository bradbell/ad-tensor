// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/adten.hpp>
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
    adten
    adfn
    adom
}

Start Recording adten_t Operations
##################################
{xrst_literal ,
    include/ad_tensor/adten.hpp
    BEGIN_START_RECORDING_1, END_START_RECORDING_1
    BEGIN_START_RECORDING_2, END_START_RECORDING_2
}

Recording
*********
This starts a recording of a function:

    range = adfn(dom_par, dom_var)

This thread must not have a recording in progress
when ``start_recording`` is called.
The range space is defined when the recording is stopped; see
:ref:`stop_recording-name` .

dom_par
*******
This is the domain parameter vector of tensors.
It this argument is (is not) present, the are no parameters in this recording.

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
    examples/adten/recording.cpp
    BEGIN_CPP, END_CPP
}

{xrst_end start_recording}
*/

vector<adten_t> adten_t::start_recording(const vector<at::Tensor>& dom_var)
{   //
    vector<at::Tensor> dom_par;
    auto [ adom_par, adom_var] = start_recording(dom_par, dom_var);
    return adom_var;
}
std::tuple< vector<adten_t>, vector<adten_t> >
adten_t::start_recording(
        const vector<at::Tensor>& dom_par ,
        const vector<at::Tensor>& dom_var )
{   //
    //
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
    // tape.m_par.m_dom_shapes
    tape.m_par.m_dom_shapes.resize( dom_par.size() );
    for(size_t index = 0; index < dom_par.size(); ++index) {
        vector<int64_t>& shape = tape.m_par.m_dom_shapes[index];
        c10::IntArrayRef sizes = dom_par[index].sizes();
        shape.resize( sizes.size() );
        for(size_t i = 0; i < sizes.size(); ++i) {
            shape[i] = sizes[i];
        }
    }
    //
    // tape.m_var.m_dom_shapes
    tape.m_var.m_dom_shapes.resize( dom_var.size() );
    for(size_t index = 0; index < dom_var.size(); ++index) {
        vector<int64_t>& shape = tape.m_var.m_dom_shapes[index];
        c10::IntArrayRef sizes = dom_var[index].sizes();
        shape.resize( sizes.size() );
        for(size_t i = 0; i < sizes.size(); ++i) {
            shape[i] = sizes[i];
        }
    }
    //
    // adom_par
    // tape.m_par: m_op_seq, m_arg_strt
    ad_type_t parameter = ad_type_t::parameter;
    vector<adten_t> adom_par;
    for(size_t index = 0; index < dom_par.size(); ++index) {
        tape.m_par.m_op_seq.push_back( dev::op_enum_t::dom );
        tape.m_par.m_arg_start.push_back( 0 );
        adom_par.push_back(
            adten_t( tape_id, index, dom_par[index], parameter)
        );
    }
    //
    // adom_var
    // tape.m_var: m_op_seq, m_arg_strt
    ad_type_t variable = ad_type_t::variable;
    vector<adten_t> adom_var;
    for(size_t index = 0; index < dom_var.size(); ++index) {
        tape.m_var.m_op_seq.push_back( dev::op_enum_t::dom );
        tape.m_var.m_arg_start.push_back( 0 );
        adom_var.push_back(
            adten_t( tape_id, index, dom_var[index], variable)
        );
    }
    return std::tuple<
        vector<adten_t>, vector<adten_t>
    > ( adom_par, adom_var);
}
/*
------------------------------------------------------------------------------
{xrst_begin stop_recording usr}
{xrst_spell
    adten
    adfn
    arange
}

Stop Recording adten_t Operations
#################################
{xrst_literal ,
    BEGIN_STOP_RECORDING, END_STOP_RECORDING
}

Recording
*********
This completes the recording of a function:

    range = adfn(dom_par, dom_var)

This thread must have a recording in progress
when ``stop_recording`` is called.
The dom_par and dom_var spaces are defined when the recording is started; see
ref:`start_recording-name` .

arange
******
This vector of AD tensors specifies the range for the AD function *adfn* .

name
****
is the name attached to the return adfn.

adfn
****
The operation sequence that was recording is transferred to this AD function.
The domain parameter and variable tensors for this function are defined
in the corresponding :ref:`start_recording-name` .

Example
*******
{xrst_literal ,
    examples/adten/recording.cpp
    BEGIN_CPP, END_CPP
}

{xrst_end stop_recording}
*/
// BEGIN_STOP_RECORDING
// adfn = adten_t::stop_recording(arnage, name)
adfn_t adten_t::stop_recording(
    const vector<adten_t>& arange ,
    const std::string&     name   )
{   // END_STOP_RECORDING
    //
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
    // adfn, tape: m_con, m_par, m_var
    adfn.m_con.swap( tape.m_con );
    adfn.m_par.swap( tape.m_par );
    adfn.m_var.swap( tape.m_var );
    //
    // adfn: m_rng_index, m_rng_adtype
    for(size_t i = 0; i < arange.size(); ++i)
    {   adfn.m_rng_index.push_back( arange[i].m_index );
        adfn.m_rng_ad_type.push_back( arange[i].m_ad_type );
    }
    //
    // adfn: m_rng_shapes
    adfn.m_rng_shapes.resize( arange.size() );
    for(size_t index = 0; index < arange.size(); ++index) {
        vector<int64_t>& shape = adfn.m_rng_shapes[index];
        c10::IntArrayRef sizes = arange[index].sizes();
        shape.resize( sizes.size() );
        for(size_t i = 0; i < sizes.size(); ++i) {
            shape[i] = sizes[i];
        }
    }
    //
    // adfn: name
    adfn.m_options.set_name(name);
    //
    return adfn;
}
// ---------------------------------------------------------------------------
} // END_NAMESPACE_AD_TENSOR
