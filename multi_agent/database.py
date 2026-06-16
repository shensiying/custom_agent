# database.py — 多 Agent 共享数据库操作
from typing import Dict, Any, Optional, List
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from config import DATABASE_URI, CURRENT_USER_ID


class OrderDB:
    def __init__(self):
        self.engine: Engine = create_engine(DATABASE_URI, echo=False)

    def _execute_query(self, query: str, params: Dict = None) -> List[Dict]:
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            columns = result.keys()
            return [dict(zip(columns, row)) for row in result.fetchall()]

    def _execute_update(self, query: str, params: Dict = None) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text(query), params or {})
                conn.commit()
                return True
        except Exception as e:
            print(f"数据库操作失败: {e}")
            return False

    def get_order_by_id(self, order_id: str, user_id: str = CURRENT_USER_ID) -> Optional[dict]:
        query = """
        SELECT id, order_id, user_id, status, amount,
               DATE_FORMAT(created_at, '%Y-%m-%d %H:%i:%s') as created_at,
               shipping_address, receiver_name, receiver_phone, logistics_status
        FROM orders
        WHERE order_id = :order_id AND user_id = :user_id
        """
        results = self._execute_query(query, {"order_id": order_id, "user_id": user_id})
        return results[0] if results else None

    def get_orders_by_user(self, user_id: str = CURRENT_USER_ID) -> List[Dict]:
        query = """
        SELECT order_id, status, amount,
               DATE_FORMAT(created_at, '%Y-%m-%d %H:%i:%s') as created_at,
               shipping_address, logistics_status
        FROM orders
        WHERE user_id = :user_id
        ORDER BY created_at DESC
        """
        return self._execute_query(query, {"user_id": user_id})

    def update_order_status(self, order_id: str, status: str, user_id: str = CURRENT_USER_ID) -> bool:
        query = "UPDATE orders SET status = :status WHERE order_id = :order_id AND user_id = :user_id"
        return self._execute_update(query, {"order_id": order_id, "status": status, "user_id": user_id})

    def update_shipping_address(self, order_id: str, new_address: str, user_id: str = CURRENT_USER_ID) -> bool:
        query = "UPDATE orders SET shipping_address = :address WHERE order_id = :order_id AND user_id = :user_id"
        return self._execute_update(query, {"order_id": order_id, "address": new_address, "user_id": user_id})

    def update_receiver_info(self, order_id: str, name: str = None, phone: str = None,
                              user_id: str = CURRENT_USER_ID) -> bool:
        updates = []
        params = {"order_id": order_id, "user_id": user_id}
        if name:
            updates.append("receiver_name = :name")
            params["name"] = name
        if phone:
            updates.append("receiver_phone = :phone")
            params["phone"] = phone
        if not updates:
            return False
        query = f"UPDATE orders SET {', '.join(updates)} WHERE order_id = :order_id AND user_id = :user_id"
        return self._execute_update(query, params)

    def order_exists(self, order_id: str, user_id: str = CURRENT_USER_ID) -> bool:
        return self.get_order_by_id(order_id, user_id) is not None


db = OrderDB()


def get_order(order_id: str, user_id: str = CURRENT_USER_ID):
    return db.get_order_by_id(order_id, user_id)

def get_orders(user_id: str = CURRENT_USER_ID):
    return db.get_orders_by_user(user_id)

def update_order(order_id: str, updates: dict, user_id: str = CURRENT_USER_ID):
    for key, value in updates.items():
        db._execute_update(
            f"UPDATE orders SET {key} = :value WHERE order_id = :order_id AND user_id = :user_id",
            {"value": value, "order_id": order_id, "user_id": user_id}
        )
    return True

def order_exists(order_id: str, user_id: str = CURRENT_USER_ID):
    return db.order_exists(order_id, user_id)
