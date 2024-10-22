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

    @llm.ai_callable(description="dapatkan produk pesanan berdasarkan status pesanan")
    def get_order_product_by_status(self, status: Annotated[str, llm.TypeInfo(description="order status")]):
        logger.info("get order product - status %s", status)

        products = self._data[self._data["Status Pemesanan"]==status.lower()]["Riwayat Pesanan"]
        logger.info(f"products {products.to_list()}")
        return f"Produk-produk dengan status pesanan {status} adalah {products.to_list()} dari {len(products)} pesanan"
    
    @llm.ai_callable(description="dapatkan produk pesanan berdasarkan metode pembayaran")
    def get_order_product_by_payment_method(self, payment_method: Annotated[str, llm.TypeInfo(description="metode pembayaran")]):
        logger.info("get order product - payment_method %s", payment_method)

        products = self._data[self._data["Metode Pembayaran"]==payment_method.lower()]["Riwayat Pesanan"]
        logger.info(f"products {products.to_list()}")
        return f"Produk-produk dengan metode pembayaran {payment_method} adalah {products.to_list()} dari {len(products)} pesanan"
    
    @llm.ai_callable(description="dapatkan produk pesanan berdasarkan id pengguna")
    def get_order_product_by_user_id(self, user_id: Annotated[int, llm.TypeInfo(description="id pengguna")]):
        logger.info("get order product - user_id %d", user_id)
        print(f"user id type", type(user_id))
        products = self._data[self._data["ID Pelanggan"]== int(user_id)]["Riwayat Pesanan"]
        logger.info(f"products {products.to_list()}")
        return f"Produk-produk dengan ID Pengguna {user_id} adalah {products.to_list()} dari {len(products)} pesanan"
        
    @llm.ai_callable(description="dapatkan data pesanan berdasarkan ID pesanan")
    def get_order_data_by_id(self, order_id: Annotated[int, llm.TypeInfo(description="id pesanan")]):
        logger.info("get order product - order_id %d", order_id)

        order = self._data[self._data["ID Pesanan"]==int(order_id)]
        logger.info(f"order {order}")
        return f"Data pesanan dengan ID pesanan {order_id} adalah {order}"