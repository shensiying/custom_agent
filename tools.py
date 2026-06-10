# tools.py
from datetime import datetime
from typing import Any, Dict, Tuple
from langchain_core.tools import tool
from database import get_order, get_orders, update_order, order_exists


def can_return(order: dict[str, Any]) -> tuple[bool, str]:
    """判断订单是否可以退货"""
    status = order.get("status", "")
    if status not in ["completed", "delivered"]:
        return False, f"订单状态为 {status}，无法退货。只有已完成或已送达的订单才能申请退货。"
    
    created_at = datetime.fromisoformat(order["created_at"]) if isinstance(order["created_at"], str) else order["created_at"]
    days_diff = (datetime.now() - created_at).days
    if days_diff > 7:
        return False, f"订单已超过7天退货期（已过{days_diff}天），无法自动退货。是否需要为您转接人工客服？"
    
    return True, "符合退货条件"

def can_exchange(order: Dict[str, Any]) -> Tuple[bool, str]:
    """判断订单是否可以换货"""
    status = order.get("status", "")
    if status not in ["completed", "delivered", "shipped"]:
        return False, f"订单状态为 {status}，无法换货。"
    
    created_at = datetime.fromisoformat(order["created_at"]) if isinstance(order["created_at"], str) else order["created_at"]
    days_diff = (datetime.now() - created_at).days
    if days_diff > 15:
        return False, f"订单已超过15天换货期（已过{days_diff}天），无法自动换货。"
    
    return True, "符合换货条件"

def can_cancel_shipment(order: Dict[str, Any]) -> Tuple[bool, str]:
    """判断订单是否可以拦截快递"""
    logistics_status = order.get("logistics_status", "")
    status = order.get("status", "")
    
    if status == "cancelled":
        return False, "订单已取消，无需重复拦截。"
    if status == "returning":
        return False, "订单已在退货流程中，无需拦截。"
    if logistics_status == "delivered":
        return False, "订单已签收，无法拦截快递。"
    if logistics_status == "shipped":
        return True, "订单已发货但未签收，可以尝试拦截。"
    if status == "pending":
        return True, "订单尚未发货，可以取消订单。"
    
    return False, f"当前状态无法拦截快递（物流状态: {logistics_status}, 订单状态: {status}）。"

def can_change_address(order: Dict[str, Any]) -> Tuple[bool, str]:
    """判断是否可以修改地址"""
    status = order.get("status", "")
    if status in ["pending", "paid"]:
        return True, ""
    return False, f"订单已 {status}，无法修改收货地址。只有待支付或已支付未发货的订单可以修改地址。"

def can_change_receiver(order: Dict[str, Any]) -> Tuple[bool, str]:
    """判断是否可以修改收件人信息"""
    return can_change_address(order)  # 与改地址规则相同

def can_query_order(order: Dict[str, Any]) -> Tuple[bool, str]:
    """查询订单始终允许"""
    return True, ""

@tool
def query_order(order_id: str) -> str:
    """
    查询订单详情。参数为6位数字订单号。
    返回订单当前状态和收货地址。
    """
    order = get_order(order_id)
    if not order:
        return f"未找到订单 {order_id}，请确认订单号是否正确。"
    return f"订单{order_id}当前状态：{order['status']}，收货地址：{order['shipping_address']}"

@tool
def return_order(order_id: str, reason: str = "未提供") -> str:
    """
    申请退货。参数为订单号和退货原因（可选）。
    规则：只有状态为'completed'且创建时间≤7天的订单可自动退货。
    """
    order = get_order(order_id)
    if not order:
        return f"未找到订单 {order_id}"
    
    # 业务规则检查
    if order["status"] != "completed":
        return f"订单状态为 {order['status']}，无法退货。已为您转人工处理。"
    
    created_at = datetime.fromisoformat(order["created_at"])
    days_diff = (datetime.now() - created_at).days
    if days_diff > 7:
        return f"订单已超过7天退货期（已过{days_diff}天），无法自动退货。是否需要人工审核？"
    
    # 执行退货
    success = update_order(order_id, {"status": "returning"})
    if success:
        return f"退货成功！订单 {order_id} 已进入退货流程。退款将在3个工作日内原路返回。"
    else:
        return f"订单 {order_id} 退货失败，请稍后重试或联系人工客服。"

@tool
def change_address(order_id: str, new_address: str) -> str:
    """
    修改订单收货地址。参数为订单号和新地址。
    规则：只有状态为'pending'或'paid'的订单可修改。
    """
    order = get_order(order_id)
    if not order:
        return f"未找到订单 {order_id}"
    
    if order["status"] not in ["pending", "paid"]:
        return f"订单已 {order['status']}，无法修改地址。如需帮助请联系人工客服。"
    
    success = update_order(order_id, {"shipping_address": new_address})
    if success:
        return f"地址修改成功！订单 {order_id} 的新地址为：{new_address}"
    else:
        return f"地址修改失败，请稍后重试。"


@tool
def query_my_orders(query: str = "") -> str:
    """
    查询当前用户的所有订单列表。可选参数 query 用于筛选（暂未实现筛选，返回全部订单）。
    """
    orders = get_orders()
    if not orders:
        return "您暂无订单记录。"
    lines = [f"订单号: {o['order_id']} | 状态: {o['status']} | 金额: {o['amount']}元 | 快递: {o.get('logistics_status', 'N/A')} | 日期: {o['created_at']}" for o in orders]
    return "\n".join(lines)


@tool
def exchange_order(order_id: str, reason: str = "") -> str:
    """
    为指定订单申请换货。order_id 为订单号，reason 为换货原因（可选）。
    规则：状态为 completed/delivered/shipped 且创建不超过15天的订单可换货。
    """
    order = get_order(order_id)
    if not order:
        return f"未找到订单 {order_id}"
    ok, msg = can_exchange(order)
    if not ok:
        return msg
    success = update_order(order_id, {"status": "exchanging"})
    if success:
        extra = f"（原因：{reason}）" if reason else ""
        return f"换货申请成功！订单 {order_id} 已进入换货流程{extra}。"
    return f"订单 {order_id} 换货申请失败，请稍后重试。"


@tool
def cancel_shipment(order_id: str) -> str:
    """
    拦截/取消指定订单的快递。order_id 为订单号。
    规则：已发货未签收的订单可尝试拦截，未发货的订单可直接取消。
    """
    order = get_order(order_id)
    if not order:
        return f"未找到订单 {order_id}"
    ok, msg = can_cancel_shipment(order)
    if not ok:
        return msg
    new_status = "cancelled" if order["status"] == "pending" else "cancelling"
    success = update_order(order_id, {"status": new_status})
    if success:
        return f"拦截/取消成功！订单 {order_id} 状态已更新为 {new_status}。"
    return f"订单 {order_id} 操作失败，请稍后重试。"


@tool
def change_receiver_info(order_id: str, name: str = "", phone: str = "") -> str:
    """
    修改指定订单的收件人信息。order_id 为订单号，name 为新收件人姓名，phone 为新电话。
    至少提供 name 或 phone 之一。
    规则：只有 pending/paid 状态的订单可修改。
    """
    order = get_order(order_id)
    if not order:
        return f"未找到订单 {order_id}"
    ok, msg = can_change_receiver(order)
    if not ok:
        return msg
    updates = {}
    if name:
        updates["receiver_name"] = name
    if phone:
        updates["receiver_phone"] = phone
    if not updates:
        return "请至少提供姓名或电话中的一个。"
    success = update_order(order_id, updates)
    if success:
        return f"收件人信息修改成功！{', '.join(f'{k}: {v}' for k, v in updates.items())}"
    return f"修改失败，请稍后重试。"