import requests
import os

def get_deerapi_balance(api_key):
    """
    查询 DeerAPI 账户余额
    参数:
        api_key (str): DeerAPI 平台注册的 API Key
    返回:
        dict: 解析后的账户数据（包含用户名、总余额、密钥详情等）
    异常:
        抛出 requests 相关异常或自定义错误
    """
    # API 配置
    url = "https://query.deerapi.com/user/quota"
    params = {"key": api_key}

    try:
        # 发送 GET 请求
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # 检查 HTTP 错误

        # 解析 JSON 响应
        data = response.json()

        # 验证关键字段是否存在
        required_fields = ["username", "total_quota", "total_used_quota", "request_count"]
        if not all(field in data for field in required_fields):
            raise ValueError("API 返回缺少关键字段")

        return data

    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"网络请求失败: {str(e)}")
    except ValueError as e:
        raise ValueError(f"JSON 解析错误: {str(e)}")

def format_balance(data):
    """
    格式化余额信息输出
    参数:
        data (dict): get_deerapi_balance() 返回的数据
    """
    # 主账户信息
    print(f"👤 用户名: {data['username']}")
    print(f"💰 总余额: ${data['total_quota']:.2f} 美元")
    print(f"🔄 累计消耗: ${data['total_used_quota']:.2f} 美元")
    print(f"📊 请求次数: {data['request_count']} 次\n")

    # 密钥详情
    print("🔑 API 密钥详情:")
    for key in data.get("keys", []):
        remain = "无限额度" if key.get("remain_quota") == -1 else f"${key['remain_quota']:.2f}"
        used = "不统计" if key.get("used_quota") == -1 else f"${key['used_quota']:.2f}"
        print(f"  ├─ {key['name']}:")
        print(f"  │   ► 剩余: {remain}")
        print(f"  │   ► 已用: {used}")

if __name__ == "__main__":
    try:
        # 从环境变量获取 API Key（推荐方式）
        # api_key = os.getenv("DEERAPI_KEY")
        api_key = "your api key"
        # 若无环境变量则手动输入
        if not api_key:
            api_key = input("请输入 DeerAPI Key: ").strip()

        if not api_key.startswith("sk-"):
            raise ValueError("无效的 API Key 格式，应以 sk- 开头")

        # 获取并显示余额
        balance_data = get_deerapi_balance(api_key)
        format_balance(balance_data)

    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        print("💡 排查建议:")
        print("  1. 检查 API Key 是否正确且以 sk- 开头")
        print("  2. 确认网络连接正常")
        print("  3. 访问 https://query.deerapi.com 验证服务状态")