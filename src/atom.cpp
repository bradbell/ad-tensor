// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include<thread>
#include<chrono>
#include <ad_tensor/atom.hpp>
#include <ad_tensor/dev/user_assert.hpp>
#include <ad_tensor/dev/move_swap.hpp>
//
namespace ad_tensor {
    // -----------------------------------------------------------------------
    // atom_global_t
    // -----------------------------------------------------------------------
    // atom_global_t::store
    size_t atom_global_t::store(
        std::unique_ptr<base_atom_t>&   base_atom_ptr) {
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
        //
        // atom_id
        size_t atom_id = m_base_vec.size();
        //
        // m_base_vec
        m_base_vec.push_back( std::make_unique<base_atom_t>() );
        dev::move_swap(m_base_vec[atom_id], base_atom_ptr);
        //
        // m_rw_mutex
        if( lock ) {
            m_rw_mutex.unlock();
        }
        //
        return atom_id;
    }
    //
    // get_base_atom
    // not const because m_rw_mutex is modified
    const base_atom_t& atom_global_t::get_base_atom(size_t atom_id) {
        std::shared_lock<std::shared_mutex> lock(m_rw_mutex);
        return *m_base_vec[atom_id];
    }
    //
    // atom_global_t::singleton
    atom_global_t& atom_global_t::singleton(void) {
        static atom_global_t atom_global;
        return atom_global;
    }
}
