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
    void atom_t::set_ ## name(const name ## _t& name) { \
        dev::user_assert( name != nullptr, \
            "atom " #name " is the nullptr in set_" #name  \
        ); \
        m_ ## name = name; \
    } \
    const atom_t::name ## _t& atom_t::get_ ## name(void) const { \
        if( m_ ## name == nullptr ) { \
            std::string msg = m_name + " atom: " #name " has not been set"; \
            dev::user_assert(false, msg); \
        } \
        return m_ ## name; \
    }
//
namespace ad_tensor {
    //
    // atom_info_t::store
    size_t atom_info_t::store(const atom_t& atom) {
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
            std::cerr << "atom_info::store: tried for 1 second to get a lock\n";
#ifndef NDEBUG
            std::exit(1);
#else
            assert(lock);
#endif
        }
        // m_atom_vec
        m_atom_vec.push_back( atom );
        //
        // m_rw_mutex
        if( lock ) {
            m_rw_mutex.unlock();
        }
        //
        // atom_id
        return m_atom_vec.size() - 1;
    }
    //
    // atom_info_t::get
    const atom_t& atom_info_t::get(size_t atom_id) {
        std::shared_lock<std::shared_mutex> lock(m_rw_mutex);
        return m_atom_vec[atom_id];
    }
    //
    // atom_t::set_name
    void atom_t::set_name(const std::string& name) {
        dev::user_assert( name != "",
            "atom set_name: name is the empty string"
        );
        m_name = name;
    }
    //
    // atom_t:: set_depend, get_depend
    SETTER_AND_GETTER(depend)
    //
    // atom_t::
    // set_forward_var, get_forward_var, set_ad_forward_var, get_ad_forward_var
    SETTER_AND_GETTER(forward_var)
    SETTER_AND_GETTER(ad_forward_var)
    //
    // atom_t::
    // set_forward_der, get_forward_der, set_ad_forward_der, get_ad_forward_der
    SETTER_AND_GETTER(forward_der)
    SETTER_AND_GETTER(ad_forward_der)
    //
    // atom_t::
    // set_reverse_der, get_reverse_der, set_ad_reverse_der, get_ad_reverse_der
    SETTER_AND_GETTER(reverse_der)
    SETTER_AND_GETTER(ad_reverse_der)
}
