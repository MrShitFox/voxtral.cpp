#ifndef VOXTRAL_SERVER_AUTH_H
#define VOXTRAL_SERVER_AUTH_H

#include <string_view>

namespace voxtral::server {

bool constant_time_equal(std::string_view left, std::string_view right);
bool authorize_bearer(
    std::string_view authorization_header,
    std::string_view configured_key,
    bool authentication_enabled);

} // namespace voxtral::server

#endif
