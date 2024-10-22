import enum
from typing import Annotated, List
from livekit.agents import llm
import logging
from dataclasses import dataclass
import datetime
import pandas as pd

logger = logging.getLogger("data-pemesanan-control")
logger.setLevel(logging.INFO)


@dataclass
class Pemesanan():
    id_pesanan: str
    status: str
    tanggal_pemesanan: datetime.datetime
    jumlah_item: int
    metode_pembayaran: str
    estimasi_pengiriman: datetime.datetime
    riwayat_pesanan: str
    id_pelanggan: str

def load_data(file_path: str) -> List[Pemesanan]:
    return pd.read_csv(file_path)


class AssistantFunc(llm.FunctionContext):
    def __init__(self, file_path:str) -> None:
        super().__init__()

        self._data = load_data(file_path)
        print("data", self._data)

    @llm.ai_callable(description="dapatkan id pesanan yang terdapat pemesanan produk tertentu")
    def get_order_by_product(self, product: Annotated[str, llm.TypeInfo(description="id pengguna")]):
        logger.info("get order id - product %s", product)
        orders = self._data[self._data["Riwayat Pesanan"].str.contains(product.title(), na=False)]["ID Pesanan"]
        logger.info(f"orders {orders.to_list()}")
        return f"Id pesanan yang memiliki produk {product} adalah {orders.to_list()} dari {len(orders)} pesanan"

    @llm.ai_callable(description="dapatkan id pesanan dengan status pesanan tertentu")
    def get_order_by_status(self, status: Annotated[str, llm.TypeInfo(description="order status")]):
        logger.info("get order id - status %s", status)
        orders = self._data[self._data["Status Pemesanan"]==status.lower()]["ID Pesanan"]
        logger.info(f"orders {orders.to_list()}")
        return f"Id pesanan yang memiliki status pesanan {status} adalah {orders.to_list()} dari {len(orders)} pesanan"
    
    @llm.ai_callable(description="dapatkan id pesanan dengan metode pembayaran tertentu")
    def get_order_by_payment_method(self, payment_method: Annotated[str, llm.TypeInfo(description="metode pembayaran")]):
        logger.info("get order id - payment_method %s", payment_method)
        orders = self._data[self._data["Metode Pembayaran"]==payment_method.lower()]["ID Pesanan"]
        logger.info(f"orders {orders.to_list()}")
        return f"Id pesanan yang memiliki metode pembayaran {payment_method} adalah {orders.to_list()} dari {len(orders)} pesanan"
    
    @llm.ai_callable(description="dapatkan id pesanan dengan id pengguna tertentu")
    def get_order_by_user_id(self, user_id: Annotated[int, llm.TypeInfo(description="id pengguna")]):
        logger.info("get order id - user_id %d", user_id)
        orders = self._data[self._data["ID Pelanggan"]== int(user_id)]["ID Pesanan"]
        logger.info(f"orders {orders.to_list()}")
        return f"Id pesanan yang memiliki ID Pengguna {user_id} adalah {orders.to_list()} dari {len(orders)} pesanan"
    
        
    @llm.ai_callable(description="dapatkan data pesanan berdasarkan ID pesanan")
    def get_order_data_by_id(self, order_id: Annotated[int, llm.TypeInfo(description="id pesanan")]):
        logger.info("get order data - order_id %d", order_id)

        order = self._data[self._data["ID Pesanan"]==int(order_id)]
        logger.info(f"order {order}")
        return f"Data pesanan dengan ID pesanan {order_id} adalah {order}"