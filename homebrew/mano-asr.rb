# Formula/mano-asr.rb
# 本地语音转写服务，基于 MLX Fun-ASR-Nano，针对 Apple Silicon 优化

class ManoAsr < Formula
  desc "本地语音转写服务，基于 MLX Fun-ASR-Nano，针对 Apple Silicon 优化"
  homepage "https://github.com/mano-asr/mano-asr"
  url "https://github.com/mano-asr/mano-asr/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "PLACEHOLDER_SOURCE_SHA256"
  license "MIT"

  bottle do
    root_url "https://github.com/mano-asr/mano-asr/releases/download/v0.1.0"
    sha256 cellar: :any_skip_relocation, arm64_tahoe: "PLACEHOLDER_BOTTLE_SHA256"
    sha256 cellar: :any_skip_relocation, arm64_sequoia: "PLACEHOLDER_BOTTLE_SHA256"
    sha256 cellar: :any_skip_relocation, arm64_sonoma: "PLACEHOLDER_BOTTLE_SHA256"
  end

  depends_on "ffmpeg"
  depends_on :macos => :monterey
  depends_on :arch => :arm64

  def install
    venv = virtualenv_create(libexec, "python3")
    venv.pip_install_and_link buildpath

    site_packages = Dir[libexec/"lib/python*/site-packages"].first
    cp_r "core", site_packages
    cp_r "utils", site_packages
    cp "server.py", site_packages

    python_version = Dir[libexec/"lib/python*"].map { |d| File.basename(d) }.first
    (bin/"mano-asr").write <<~EOS
      #!/bin/bash
      exec "#{libexec}/bin/#{python_version}" -m manoasr.cli.main "$@"
    EOS
    chmod 0755, bin/"mano-asr"
  end

  def caveats
    <<~EOS
      mano-asr 安装完成！

      模型将在首次运行时自动下载（约 1-2 GB）。

      快速开始:
        mano-asr start              # 启动服务（首次自动下载模型）
        mano-asr transcribe a.wav   # 转写音频
        mano-asr model              # 切换 ASR 引擎

      服务管理:
        mano-asr start / stop / restart / status
        mano-asr logs --stats       # 查看统计
        mano-asr doctor             # 环境检查

      模型存储: ~/.mano-asr/models/
      服务地址: http://127.0.0.1:8787
    EOS
  end

  service do
    run [opt_bin/"mano-asr", "start", "--foreground"]
    keep_alive true
    working_dir var
    log_path var/"log/mano-asr.log"
    error_log_path var/"log/mano-asr.log"
  end

  test do
    assert_match "0.1.0", shell_output("#{bin}/mano-asr --version")
  end
end
