// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/sparsity.hpp>
#include <algorithm>

namespace ad_tensor {
    //
    // size
    size_t sparsity_t::size(void) {
        return m_sparsity.size();
    }
    // sort
    void sparsity_t::sort(void) {
        std::sort( m_sparsity.begin(), m_sparsity.end() );
    }
    //
    // push_back
    void sparsity_t::push_back( const std::array<size_t, 2>& pair ) {
        m_sparsity.push_back( pair );
    }
    //
    // operator[]
    const std::array<size_t, 2>& sparsity_t::operator[](size_t index) const {
        return m_sparsity[index];
    }
}
