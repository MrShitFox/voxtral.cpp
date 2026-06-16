# Homebrew formula for voxtral.cpp.
#
#   brew tap andrijdavid/voxtral https://github.com/andrijdavid/voxtral.cpp
#   brew install --HEAD voxtral
#
# Builds against the system ggml (no bundled submodule), like the homebrew-core
# whisper-cpp formula.
class Voxtral < Formula
  desc "C++ implementation of the Voxtral speech-to-text model (ggml)"
  homepage "https://github.com/andrijdavid/voxtral.cpp"
  license "MIT"
  # Add `url` + `sha256` for a tagged release tarball; HEAD-only for now.
  head "https://github.com/andrijdavid/voxtral.cpp.git", branch: "main"

  depends_on "cmake" => :build
  depends_on "ggml"

  def install
    system "cmake", "-S", ".", "-B", "build",
           "-DVOXTRAL_USE_SYSTEM_GGML=ON",
           "-DCMAKE_INSTALL_RPATH=#{rpath}",
           *std_cmake_args
    system "cmake", "--build", "build"
    system "cmake", "--install", "build"
  end

  test do
    assert_match "usage", shell_output("#{bin}/voxtral --help")
  end
end
