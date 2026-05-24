// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
#include <ad_tensor/dev/tape.hpp>

namespace ad_tensor { namespace devel {
    tape_t& this_threads_tape(void) {
        thread_local tape_t tape;
        return tape;
    }
} }
