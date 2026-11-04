from ctypes import sizeof
from getpass import getpass

from NetSDK.NetSDK import NetClient
from NetSDK.SDK_Enum import EM_LOGIN_SPAC_CAP_TYPE
from NetSDK.SDK_Struct import (
    NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY,
    NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY,
)

CAMERA_IP = "5.202.196.201"
CAMERA_PORT = 37777

def main():
    username = input("Camera Username: ").strip()
    password = getpass("Camera Password: ")

    sdk = NetClient()
    initialized = False
    login_id = 0

    try:
        print("\nInitializing Dahua NetSDK...")

        init_result = sdk.InitEx()
        print("SDK INIT RESULT:", init_result)

        if init_result is False or init_result == 0:
            raise RuntimeError("Dahua SDK initialization failed")

        initialized = True

        login_in = NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY()
        login_in.dwSize = sizeof(NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY)
        login_in.szIP = CAMERA_IP.encode("ascii")
        login_in.nPort = CAMERA_PORT
        login_in.szUserName = username.encode("utf-8")
        login_in.szPassword = password.encode("utf-8")
        login_in.emSpecCap = EM_LOGIN_SPAC_CAP_TYPE.TCP
        login_in.pCapParam = None

        login_out = NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY()
        login_out.dwSize = sizeof(NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY)

        print("\nConnecting to Dahua device...")

        login_id, device_info, error_msg = (
            sdk.LoginWithHighLevelSecurity(login_in, login_out)
        )

        if not login_id:
            print("\nLOGIN: FAILED")
            print("ERROR:", error_msg)
            print("SDK ERROR:", sdk.GetLastErrorMessage())
            return

        print("\nLOGIN: SUCCESS")
        print("DEVICE IP:", CAMERA_IP)
        print("DEVICE PORT:", CAMERA_PORT)
        print("CHANNEL COUNT:", device_info.nChanNum)

        print("\nDAHUA LOGIN TEST: PASS")

    except Exception as exc:  # noqa: BLE001
        print("\nDAHUA LOGIN TEST: ERROR")
        print(type(exc).__name__, str(exc))

    finally:
        if login_id:
            print("\nLogging out...")
            sdk.Logout(login_id)

        if initialized:
            sdk.Cleanup()

        print("SDK CLEANUP: COMPLETE")


if __name__ == "__main__":
    main()
