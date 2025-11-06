#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <DbgHelp.h>
#pragma comment(lib, "Dbghelp.lib")
#endif
#include <csignal>
#include <cstdio>
#include <ctime>
#include <string>
#include <sstream>
#include "logger.h"
#include "crash_handler.h"

namespace coacd {

#ifdef _WIN32
static std::string timestamp() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_s(&tm, &t);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
    return std::string(buf);
}

static LONG WINAPI seh_handler(EXCEPTION_POINTERS* info) {
    std::string dir = "crash_dumps";
    CreateDirectoryA(dir.c_str(), nullptr);
    std::ostringstream oss;
    oss << dir << "\\coacd_" << timestamp() << ".dmp";
    std::string path = oss.str();

    HANDLE hFile = CreateFileA(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile != INVALID_HANDLE_VALUE) {
        MINIDUMP_EXCEPTION_INFORMATION mdei;
        mdei.ThreadId = GetCurrentThreadId();
        mdei.ExceptionPointers = info;
        mdei.ClientPointers = FALSE;
        MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), hFile, MiniDumpWithFullMemory, &mdei, nullptr, nullptr);
        CloseHandle(hFile);
        logger::critical("Unhandled exception 0x{:X}. Minidump: {}", (unsigned long)info->ExceptionRecord->ExceptionCode, path);
    } else {
        logger::critical("Unhandled exception 0x{:X}. Failed to write minidump", (unsigned long)info->ExceptionRecord->ExceptionCode);
    }
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

static void sig_handler(int sig) {
    logger::critical("Caught signal {}", sig);
    std::signal(sig, SIG_DFL);
}

void install_crash_handler() {
#ifdef _WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    SetUnhandledExceptionFilter(seh_handler);
#endif
    std::signal(SIGABRT, sig_handler);
    std::signal(SIGSEGV, sig_handler);
}

} // namespace coacd
