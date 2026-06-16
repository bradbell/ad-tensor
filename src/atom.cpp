// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include<thread>
#include<chrono>
#include <ad_tensor/atom.hpp>
#include <ad_tensor/dev/user_assert.hpp>
//
#define SETTER_AND_GETTER(name) \
    void atom_callback_t::set_ ## name(const name ## _t& name) { \
        dev::user_assert( name != nullptr, \
            "atom " #name " is the nullptr in set_" #name  \
        ); \
        m_ ## name = name; \
    } \
    const atom_callback_t::name ## _t& \
    atom_callback_t::get_ ## name(void) const { \
        if( m_ ## name == nullptr ) { \
            std::string msg = m_name + \
                " atom_callback: " #name " has not been set"; \
            dev::user_assert(false, msg); \
        } \
        return m_ ## name; \
    }
//
namespace ad_tensor {
    //
    // atom_global_t::store
    size_t atom_global_t::store(const atom_callback_t& atom_callback) {
        //
        // lock, m_rw_mutex
        size_t count = 0;
        bool   lock  = false;
        while( count < 100 && ! lock ) {
            lock = m_rw_mutex.try_lock();
            if( ! lock )
            {   ++count;
                std::this_thread::sleep_for( std::chrono::milliseconds(10) );
            }
        }
        if( ! lock ) {
            std::cerr << "atom_global::store: "
                " tried for 1 second to get a lock\n";
#ifndef NDEBUG
            std::exit(1);
#else
            assert(lock);
#endif
        }
        // m_atom_callback_vec
        m_atom_callback_vec.push_back( atom_callback );
        //
        // m_rw_mutex
        if( lock ) {
            m_rw_mutex.unlock();
        }
        //
        // atom_id
        return m_atom_callback_vec.size() - 1;
    }
    //
    // atom_global_t::get
    const atom_callback_t& atom_global_t::get(size_t atom_id) {
        std::shared_lock<std::shared_mutex> lock(m_rw_mutex);
        return m_atom_callback_vec[atom_id];
    }
    //
    // atom_callback_t::set_name
    void atom_callback_t::set_name(const std::string& name) {
        dev::user_assert( name != "",
            "atom_callback set_name: name is the empty string"
        );
        m_name = name;
    }
    //
    // atom_callback_t:: set_depend, get_depend
    SETTER_AND_GETTER(depend)
    //
    // atom_callback_t::
    // set_forward, get_forward, set_ad_forward, get_ad_forward
    SETTER_AND_GETTER(forward)
    SETTER_AND_GETTER(ad_forward)
    //
    // atom_callback_t::
    // set_forward_der, get_forward_der, set_ad_forward_der, get_ad_forward_der
    SETTER_AND_GETTER(forward_der)
    SETTER_AND_GETTER(ad_forward_der)
    //
    // atom_callback_t::
    // set_reverse_der, get_reverse_der, set_ad_reverse_der, get_ad_reverse_der
    SETTER_AND_GETTER(reverse_der)
    SETTER_AND_GETTER(ad_reverse_der)
}
