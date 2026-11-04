import argparse
import time
from ctypes import sizeof
from getpass import getpass
from threading import Lock

from NetSDK.NetSDK import NetClient
from NetSDK.SDK_Callback import fRealDataCallBackEx2
from NetSDK.SDK_Enum import (
    EM_LOGIN_SPAC_CAP_TYPE,
    EM_REALDATA_FLAG,
    SDK_RealPlayType,
)
from NetSDK.SDK_Struct import (
    NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY,
    NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="5.202.196.201")
    parser.add_argument("--port", type=int, default=37777)
    parser.add_argument("--channel", type=int, default=2)
    parser.add_argument("--seconds", type=int, default=15)
    args = parser.parse_args()

    username = input("Camera Username: ").strip()
    password = getpass("Camera Password: ")

    sdk = NetClient()
    login_id = 0
    play_id = 0
    initialized = False

    stats = {"packets": 0, "bytes": 0}
    lock = Lock()

    def on_data(
        real_handle,
        data_type,
        buffer,
        buffer_size,
        param,
        user_data,
    ):
        if buffer_size > 0:
            with lock:
                stats["packets"] += 1
                stats["bytes"] += buffer_size

    callback = fRealDataCallBackEx2(on_data)

    try:
        if not sdk.InitEx():
            raise RuntimeError("SDK initialization failed")

        initialized = True

        login_in = NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY()
        login_in.dwSize = sizeof(login_in)
        login_in.szIP = args.ip.encode("ascii")
        login_in.nPort = args.port
        login_in.szUserName = username.encode("utf-8")
        login_in.szPassword = password.encode("utf-8")
        login_in.emSpecCap = EM_LOGIN_SPAC_CAP_TYPE.TCP
        login_in.pCapParam = None

        login_out = NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY()
        login_out.dwSize = sizeof(login_out)

        login_id, device_info, error = (
            sdk.LoginWithHighLevelSecurity(
                login_in,
                login_out,
            )
        )

        if not login_id:
            raise RuntimeError(f"Login failed: {error}")

        channel_count = int(device_info.nChanNum)

        print("LOGIN: PASS")
        print("CHANNEL COUNT:", channel_count)

        if not 1 <= args.channel <= channel_count:
            raise ValueError("Channel number is out of range")

        # Dahua SDK channels are zero-based.
        channel_index = args.channel - 1

        print("DMSS CHANNEL:", args.channel)
        print("SDK CHANNEL INDEX:", channel_index)

        play_id = sdk.RealPlayEx(
            login_id,
            channel_index,
            0,
            SDK_RealPlayType.Realplay,
        )

        if not play_id:
            raise RuntimeError(
                "RealPlay failed: "
                + sdk.GetLastErrorMessage()
            )

        registered = sdk.SetRealDataCallBackEx2(
            play_id,
            callback,
            None,
            EM_REALDATA_FLAG.RAW_DATA,
        )

        if not registered:
            raise RuntimeError(
                "Callback registration failed: "
                + sdk.GetLastErrorMessage()
            )

        print("REALPLAY: STARTED")
        print("RECEIVING VIDEO DATA...")

        time.sleep(args.seconds)

        with lock:
            packets = stats["packets"]
            total_bytes = stats["bytes"]

        print("PACKETS RECEIVED:", packets)
        print("TOTAL BYTES:", total_bytes)

        if packets > 0 and total_bytes > 0:
            print("DAHUA VIDEO DATA TEST: PASS")
        else:
            print("DAHUA VIDEO DATA TEST: FAIL")

    except Exception as exc:  # noqa: BLE001
        print("TEST ERROR:", type(exc).__name__, str(exc))

    finally:
        if play_id:
            sdk.StopRealPlayEx(play_id)
            print("REALPLAY: STOPPED")

        if login_id:
            sdk.Logout(login_id)
            print("LOGOUT: COMPLETE")

        if initialized:
            sdk.Cleanup()
            print("SDK CLEANUP: COMPLETE")


if __name__ == "__main__":
    main()
