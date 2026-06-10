// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
# include <ad_tensor/options.hpp>
//
namespace ad_tensor {
    void options_t::set_trace(bool trace) {
        m_trace = trace;
    }
    bool options_t::get_trace(void) const {
        return m_trace;
    }
}
