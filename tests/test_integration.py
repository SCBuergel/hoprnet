import asyncio
import itertools
import random
import re
from contextlib import AsyncExitStack, asynccontextmanager

import pytest
from conftest import (
    DEFAULT_API_TOKEN,
    NODES,
    OPEN_CHANNEL_FUNDING_VALUE,
    TICKET_AGGREGATION_THRESHOLD,
    TICKET_PRICE_PER_HOP,
)

PARAMETERIZED_SAMPLE_SIZE = 1  # if os.getenv("CI", default="false") == "false" else 3
AGGREGATED_TICKET_PRICE = TICKET_AGGREGATION_THRESHOLD * TICKET_PRICE_PER_HOP
MULTIHOP_MESSAGE_SEND_TIMEOUT = 30.0
CHECK_RETRY_INTERVAL = 0.5


def shuffled(coll):
    random.shuffle(coll)
    return coll


def default_nodes():
    """All nodes within the same network as specified in the swarm7 fixture"""
    return list(NODES.keys())[:4]


def passive_node():
    """A node that uses no strategy"""
    return "5"


@asynccontextmanager
async def create_channel(src, dest, funding: int):
    channel = await src["api"].open_channel(dest["address"], str(int(funding)))
    assert channel is not None
    await asyncio.wait_for(check_channel_status(src, dest, status="Open"), 10.0)
    try:
        yield channel
    finally:
        assert await src["api"].close_channel(channel)
        await asyncio.wait_for(check_channel_status(src, dest, status="PendingToClose"), 10.0)

        # need to wait some more time until closure time has passed
        await asyncio.sleep(15)

        assert await src["api"].close_channel(channel)
        await asyncio.wait_for(check_channel_status(src, dest, status="Closed"), 10.0)


async def get_channel(src, dest, include_closed=False):
    open_channels = await src["api"].all_channels(include_closed=include_closed)
    channels = [
        oc
        for oc in open_channels.all
        if oc.source_address == src["address"] and oc.destination_address == dest["address"]
    ]

    return channels[0] if len(channels) > 0 else None


async def get_channel_seen_from_dst(src, dest, include_closed=False):
    open_channels = await dest["api"].all_channels(include_closed=include_closed)
    channels = [
        oc
        for oc in open_channels.all
        if oc.source_address == src["address"] and oc.destination_address == dest["address"]
    ]

    return channels[0] if len(channels) > 0 else None


async def check_channel_status(src, dest, status):
    assert status in ["Open", "PendingToClose", "Closed"]
    while True:
        channel = await get_channel(src, dest, include_closed=False)
        channel_seen_from_dst = await get_channel_seen_from_dst(src, dest, include_closed=False)
        if (
            channel is not None
            and channel.status == status
            and channel_seen_from_dst is not None
            and channel_seen_from_dst.status == status
        ):
            break
        else:
            await asyncio.sleep(CHECK_RETRY_INTERVAL)


async def check_outgoing_channel_closed(src, channel_id: str):
    while True:
        channel = await src["api"].get_channel(channel_id)
        if channel is not None and channel.status == "Closed":
            break
        else:
            await asyncio.sleep(CHECK_RETRY_INTERVAL)


async def check_received_packets_with_pop(receiver, expected_packets, tag=None, sort=True):
    received = []

    while len(received) != len(expected_packets):
        packet = await receiver["api"].messages_pop(tag)
        if packet is not None:
            received.append(packet.body)
        else:
            await asyncio.sleep(CHECK_RETRY_INTERVAL)

    if sort:
        expected_packets.sort()
        received.sort()

    assert received == expected_packets


async def check_received_packets_with_peek(receiver, expected_packets: list[str], tag=None, sort=True):
    received = []

    while len(received) != len(expected_packets):
        packets = await receiver["api"].messages_peek_all(tag)

        if packets is None:
            await asyncio.sleep(CHECK_RETRY_INTERVAL)
            continue

        received = [m.body for m in packets.messages]

    if sort:
        expected_packets.sort()
        received.sort()

    assert received == expected_packets, f"Expected: {expected_packets}, got: {received}"


async def check_rejected_tickets(src, count):
    while int((await src["api"].get_tickets_statistics()).rejected) < count:
        await asyncio.sleep(CHECK_RETRY_INTERVAL)


async def check_unredeemed_tickets_value(src, value):
    while balance_str_to_int((await src["api"].get_tickets_statistics()).unredeemed_value) < value:
        await asyncio.sleep(CHECK_RETRY_INTERVAL)


async def check_all_tickets_redeemed(src, channel_id=None):
    if channel_id is not None:
        while len(await src["api"].channel_get_tickets(channel_id)) > 0:
            await asyncio.sleep(CHECK_RETRY_INTERVAL)
    else:
        while (await src["api"].get_tickets_statistics()).unredeemed > 0:
            await asyncio.sleep(CHECK_RETRY_INTERVAL)


async def send_and_receive_packets_with_pop(packets, src, dest, path, timeout=MULTIHOP_MESSAGE_SEND_TIMEOUT):
    random_tag = random.randint(10, 65530)

    for packet in packets:
        assert await src["api"].send_message(dest["peer_id"], packet, path, random_tag)

    await asyncio.wait_for(check_received_packets_with_pop(dest, packets, tag=random_tag, sort=True), timeout)


async def send_and_receive_packets_with_peek(packets, src, dest, path, timeout=MULTIHOP_MESSAGE_SEND_TIMEOUT):
    random_tag = random.randint(10, 65530)

    for packet in packets:
        assert await src["api"].send_message(dest["peer_id"], packet, path, random_tag)

    await asyncio.wait_for(check_received_packets_with_peek(dest, packets, tag=random_tag, sort=True), timeout)

    return random_tag


def random_distinct_pairs_from(values: list, count: int):
    return random.sample([(left, right) for left, right in itertools.product(values, repeat=2) if left != right], count)


def balance_str_to_int(balance: str):
    return int(balance.split(" ", 1)[0])


# NOTE: this test is first, ensuring that all tests following it have ensured connectivity and
# correct ticket price from api@pytest.mark.asyncio
async def test_hoprd_swarm_connectivity(swarm7):
    async def check_all_connected(me, others: list):
        others = set(others)
        while True:
            current_peers = set([x["peer_id"] for x in await me["api"].peers()])
            if current_peers.intersection(others) == others:
                break
            else:
                current_peers.intersection(others) == others
                await asyncio.sleep(0.5)

    await asyncio.gather(
        *[
            asyncio.wait_for(
                check_all_connected(swarm7[k], [swarm7[v]["peer_id"] for v in default_nodes() if v != k]), 60.0
            )
            for k in default_nodes()
        ]
    )

    ticket_price = await random.choice(list(swarm7.values()))["api"].ticket_price()
    if ticket_price is not None:
        global TICKET_PRICE_PER_HOP, AGGREGATED_TICKET_PRICE
        TICKET_PRICE_PER_HOP = ticket_price
        AGGREGATED_TICKET_PRICE = TICKET_AGGREGATION_THRESHOLD * TICKET_PRICE_PER_HOP
    else:
        print("Could not get ticket price from API, using default value")


@pytest.mark.asyncio
async def test_hoprd_protocol_check_balances_without_prior_tests(swarm7):
    for _, node_args in swarm7.items():
        addr = await node_args["api"].addresses("native")
        assert re.match("^0x[0-9a-fA-F]{40}$", addr) is not None
        balances = await node_args["api"].balances()
        native_balance = int(balances.native.split(" ")[0])
        hopr_balance = int(balances.safe_hopr.split(" ")[0])
        assert native_balance > 0
        assert hopr_balance > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("peer", random.sample(default_nodes(), 1))
async def test_hoprd_node_should_be_able_to_alias_other_peers(peer, swarm7):
    peer_id = swarm7[random.choice(default_nodes())]["peer_id"]

    assert await swarm7[peer]["api"].aliases_set_alias("Alice", peer_id) == True
    assert await swarm7[peer]["api"].aliases_set_alias("Alice", peer_id)
    assert await swarm7[peer]["api"].aliases_get_alias("Alice") == peer_id

    assert await swarm7[peer]["api"].aliases_remove_alias("Alice")
    assert await swarm7[peer]["api"].aliases_get_alias("Alice") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("src,dest", random_distinct_pairs_from(default_nodes(), count=PARAMETERIZED_SAMPLE_SIZE))
async def test_hoprd_ping_should_work_between_nodes_in_the_same_network(src, dest, swarm7):
    response = await swarm7[src]["api"].ping(swarm7[dest]["peer_id"])

    assert response is not None
    assert int(response.latency) > 0, f"Non-0 round trip time expected, actual: '{int(response.latency)}'"


@pytest.mark.asyncio
@pytest.mark.parametrize("peer", random.sample(default_nodes(), 1))
async def test_hoprd_ping_should_timeout_on_pinging_self(peer, swarm7):
    response = await swarm7[peer]["api"].ping(swarm7[peer]["peer_id"])

    assert response is None, f"Pinging self should produce timeout, not '{response}'"


@pytest.mark.asyncio
async def test_hoprd_ping_should_not_be_able_to_ping_nodes_in_other_network_UNFINISHED(swarm7):
    """
    # FIXME: re-enable when network check works
    # log "Node 1 should not be able to talk to Node 6 (different network id)"
    # result=$(api_ping "${api6}" ${addr1} "TIMEOUT")
    # log "-- ${result}"

    # FIXME: re-enable when network check works
    # log "Node 6 should not be able to talk to Node 1 (different network id)"
    # result=$(api_ping "${api6}" ${addr1} "TIMEOUT")
    # log "-- ${result}"
    """
    assert True


@pytest.mark.asyncio
async def test_hoprd_ping_should_not_be_able_to_ping_nodes_not_present_in_the_registry_UNFINISHED(swarm7):
    """
    # log "Node 7 should not be able to talk to Node 1 (Node 7 is not in the register)"
    # result=$(ping "${api7}" ${addr1} "TIMEOUT")
    # log "-- ${result}"

    # log "Node 1 should not be able to talk to Node 7 (Node 7 is not in the register)"
    # result=$(ping "${api1}" ${addr7} "TIMEOUT")
    # log "-- ${result}"
    """
    assert True


@pytest.mark.asyncio
@pytest.mark.parametrize("peer", default_nodes())
async def test_hoprd_should_not_have_unredeemed_tickets_without_sending_messages(peer, swarm7):
    statistics = await swarm7[peer]["api"].get_tickets_statistics()

    assert balance_str_to_int(statistics.unredeemed_value) == 0
    assert int(statistics.unredeemed) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("src,dest", random_distinct_pairs_from(default_nodes(), count=PARAMETERIZED_SAMPLE_SIZE))
async def test_hoprd_should_be_able_to_send_0_hop_messages_without_open_channels(src, dest, swarm7):
    message_count = int(TICKET_AGGREGATION_THRESHOLD / 10)

    packets = [f"0 hop message #{i:08d}" for i in range(message_count)]
    await send_and_receive_packets_with_pop(packets, src=swarm7[src], dest=swarm7[dest], path=[])


@pytest.mark.asyncio
@pytest.mark.parametrize("src,dest", random_distinct_pairs_from(default_nodes(), count=PARAMETERIZED_SAMPLE_SIZE))
async def test_peeking_messages_with_tag(src, dest, swarm7):
    message_count = int(TICKET_AGGREGATION_THRESHOLD / 10)

    packets = [f"0 hop message #{i:08d}" for i in range(message_count)]
    random_tag = await send_and_receive_packets_with_peek(packets, src=swarm7[src], dest=swarm7[dest], path=[])

    # after checking that messages are in the destination inbox, a second check should again assert
    # that they are there. This shows that peeking does not remove the messages from the inbox
    await asyncio.wait_for(
        check_received_packets_with_peek(
            swarm7[dest],
            packets,
            tag=random_tag,
            sort=True,
        ),
        MULTIHOP_MESSAGE_SEND_TIMEOUT,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "src,dest", [(passive_node(), random.choice(default_nodes())) for _ in range(PARAMETERIZED_SAMPLE_SIZE)]
)
async def test_hoprd_api_channel_should_register_fund_increase_using_fund_endpoint(src, dest, swarm7):
    hopr_amount = OPEN_CHANNEL_FUNDING_VALUE

    async with create_channel(swarm7[src], swarm7[dest], funding=TICKET_PRICE_PER_HOP) as channel:
        balance_before = await swarm7[src]["api"].balances()

        assert await swarm7[src]["api"].channels_fund_channel(channel, hopr_amount)

        balance_after = await swarm7[src]["api"].balances()

        assert balance_str_to_int(balance_before.safe_hopr) - balance_str_to_int(
            balance_after.safe_hopr
        ) == balance_str_to_int(hopr_amount)
        assert balance_str_to_int(balance_before.safe_hopr_allowance) - balance_str_to_int(
            balance_after.safe_hopr_allowance
        ) == balance_str_to_int(hopr_amount)
        assert balance_str_to_int(balance_after.native) < balance_str_to_int(balance_before.native)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "src,dest", [(random.choice(default_nodes()), passive_node()) for _ in range(PARAMETERIZED_SAMPLE_SIZE)]
)
async def test_hoprd_api_should_redeem_tickets_in_channel_using_redeem_endpoint(src, dest, swarm7):
    message_count = 2

    async with create_channel(swarm7[src], swarm7[dest], funding=message_count * TICKET_PRICE_PER_HOP) as channel:
        packets = [f"Channel redeem on 1-hop: {src} - {dest} - {src} #{i:08d}" for i in range(message_count)]

        await send_and_receive_packets_with_pop(
            packets, src=swarm7[src], dest=swarm7[src], path=[swarm7[dest]["peer_id"]]
        )

        await asyncio.wait_for(check_unredeemed_tickets_value(swarm7[dest], message_count * TICKET_PRICE_PER_HOP), 30.0)

        async def channel_redeem_tickets(api, channel):
            while True:
                if await api.channel_redeem_tickets(channel):
                    break
                else:
                    await asyncio.sleep(0.5)

        await asyncio.wait_for(channel_redeem_tickets(swarm7[dest]["api"], channel), 20.0)

        await asyncio.wait_for(check_all_tickets_redeemed(swarm7[dest]), 120.0)

        assert await swarm7[dest]["api"].channel_get_tickets(channel) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "src,dest", [(passive_node(), random.choice(default_nodes())) for _ in range(PARAMETERIZED_SAMPLE_SIZE)]
)
async def test_hoprd_should_fail_sending_a_message_when_the_channel_is_out_of_funding(src, dest, swarm7):
    """
    # FIXME: The following part can be enabled once incoming channel closure is
    # implemented.
    #
    # need to close the incoming side to not have to wait for the closure timeout
    # api_close_channel "${second_node_id}" "${node_id}" "${second_node_api}" "${node_addr}" "incoming"

    # only fund for 2 tickets
    # channel_info=$(api_open_channel "${node_id}" "${second_node_id}" "${node_api}" "${second_node_addr}" "200")

    # need to wait a little to allow the other side to index the channel open event
    # sleep 10
    # api_get_tickets_in_channel ${second_node_api} ${channel_id} "TICKETS_NOT_FOUND"
    # for i in `seq 1 ${generated_tickets}`; do
    #   log "PendingBalance in channel: Node ${node_id} send 1 hop message to self via node ${second_node_id}"
    #   api_send_message "${node_api}" "${msg_tag}" "${peer_id}" \
    #       "pendingbalance: hello, world 1 self" "${second_peer_id}"
    # done

    # seems like there's slight delay needed for tickets endpoint to return up to date tickets, \
    #       probably because of blockchain sync delay
    # sleep 5

    # ticket_amount=$(api_get_tickets_in_channel ${second_node_api} ${channel_id} | jq '. | length')
    # if [[ "${ticket_amount}" != "${generated_tickets}" ]]; then
    #   msg "PendingBalance: Ticket amount ${ticket_amount} is different than expected ${generated_tickets}"
    #   exit 1
    # fi

    # api_redeem_tickets_in_channel ${second_node_api} ${channel_id}
    # sleep 5
    # api_get_tickets_in_channel ${second_node_api} ${channel_id} "TICKETS_NOT_FOUND"
    # api_close_channel "${node_id}" "${second_node_id}" "${node_api}" "${second_node_addr}" "outgoing"
    """

    message_count = 2

    async with AsyncExitStack() as channels:
        await asyncio.gather(
            *[
                channels.enter_async_context(
                    create_channel(swarm7[i[0]], swarm7[i[1]], funding=message_count * TICKET_PRICE_PER_HOP)
                )
                for i in [[src, dest], [dest, src]]
            ]
        )

        packets = [f"Channel agg and redeem on 1-hop: {src} - {dest} - {src} #{i:08d}" for i in range(message_count)]
        await send_and_receive_packets_with_pop(
            packets, src=swarm7[src], dest=swarm7[src], path=[swarm7[dest]["peer_id"]]
        )

        # this message has no funding in the channel, but it still should be sent
        assert await swarm7[src]["api"].send_message(
            swarm7[src]["peer_id"], "THIS MSG IS NOT COVERED", [swarm7[dest]["peer_id"]]
        )

        await asyncio.wait_for(check_unredeemed_tickets_value(swarm7[dest], message_count * TICKET_PRICE_PER_HOP), 30.0)

        # we should see last the message as rejected
        await asyncio.wait_for(check_rejected_tickets(swarm7[dest], 1), 120.0)

        await asyncio.sleep(10)  # wait for aggregation to finish
        assert await swarm7[dest]["api"].tickets_redeem()

        await asyncio.wait_for(check_all_tickets_redeemed(swarm7[dest]), 120.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "src,dest", [(random.choice(default_nodes()), passive_node()) for _ in range(PARAMETERIZED_SAMPLE_SIZE)]
)
async def test_hoprd_should_create_redeemable_tickets_on_routing_in_1_hop_to_self_scenario(src, dest, swarm7):
    # send 90% of messages before ticket aggregation would kick in
    message_count = int(TICKET_AGGREGATION_THRESHOLD / 10 * 9)

    async with create_channel(swarm7[src], swarm7[dest], funding=message_count * TICKET_PRICE_PER_HOP) as channel_id:
        # ensure ticket stats are what we expect before starting
        statistics_before = await swarm7[dest]["api"].get_tickets_statistics()
        tickets_before = await swarm7[dest]["api"].channel_get_tickets(channel_id)
        assert len(tickets_before) == 0

        packets = [
            f"1 hop message to self: {src} - {dest} - {src} #{i:08d} of #{message_count:08d}"
            for i in range(message_count)
        ]
        await send_and_receive_packets_with_pop(
            packets, src=swarm7[src], dest=swarm7[src], path=[swarm7[dest]["peer_id"]], timeout=60.0
        )

        await asyncio.wait_for(check_unredeemed_tickets_value(swarm7[dest], message_count * TICKET_PRICE_PER_HOP), 30.0)

        # ensure ticket stats are updated after messages are sent
        statistics_after = await swarm7[dest]["api"].get_tickets_statistics()
        tickets_after = await swarm7[dest]["api"].channel_get_tickets(channel_id)
        assert statistics_after.redeemed == statistics_before.redeemed
        assert (statistics_after.unredeemed - statistics_before.unredeemed) == len(packets)
        assert len(tickets_after) == len(packets)

        assert await swarm7[dest]["api"].channel_redeem_tickets(channel_id)

        await asyncio.wait_for(check_all_tickets_redeemed(swarm7[dest], channel_id), 120.0)

        # ensure ticket stats are updated after redemption
        statistics_after_redemption = await swarm7[dest]["api"].get_tickets_statistics()
        assert (statistics_after_redemption.redeemed - statistics_after.redeemed) == len(packets)
        assert statistics_after_redemption.unredeemed == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "src,dest", [(random.choice(default_nodes()), passive_node()) for _ in range(PARAMETERIZED_SAMPLE_SIZE)]
)
async def test_hoprd_should_aggregate_and_redeem_tickets_in_channel_on_api_request(src, dest, swarm7):
    message_count = 2

    async with create_channel(swarm7[src], swarm7[dest], funding=message_count * TICKET_PRICE_PER_HOP) as channel:
        packets = [f"Channel agg and redeem on 1-hop: {src} - {dest} - {src} #{i:08d}" for i in range(message_count)]
        await send_and_receive_packets_with_pop(
            packets, src=swarm7[src], dest=swarm7[src], path=[swarm7[dest]["peer_id"]]
        )

        await asyncio.wait_for(check_unredeemed_tickets_value(swarm7[dest], message_count * TICKET_PRICE_PER_HOP), 30.0)

        assert len(await swarm7[dest]["api"].channel_get_tickets(channel)) == 2

        async def channel_aggregate_tickets(api, target_channel):
            while True:
                if await api.channels_aggregate_tickets(target_channel):
                    break
                else:
                    await asyncio.sleep(0.5)

        await asyncio.wait_for(channel_aggregate_tickets(swarm7[dest]["api"], channel), 20.0)

        assert len(await swarm7[dest]["api"].channel_get_tickets(channel)) == 1

        assert await swarm7[dest]["api"].channel_redeem_tickets(channel)

        await asyncio.wait_for(check_all_tickets_redeemed(swarm7[dest]), 120.0)

        assert len(await swarm7[dest]["api"].channel_get_tickets(channel)) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route",
    [shuffled(default_nodes())[:3] for _ in range(PARAMETERIZED_SAMPLE_SIZE)]
    # + [shuffled(nodes())[:5] for _ in range(PARAMETERIZED_SAMPLE_SIZE)],
)
async def test_hoprd_should_create_redeemable_tickets_on_routing_in_general_n_hop(route, swarm7):
    message_count = int(TICKET_AGGREGATION_THRESHOLD / 10)

    async with AsyncExitStack() as channels:
        await asyncio.gather(
            *[
                channels.enter_async_context(
                    create_channel(swarm7[route[i]], swarm7[route[i + 1]], funding=message_count * TICKET_PRICE_PER_HOP)
                )
                for i in range(len(route) - 1)
            ]
        )

        packets = [f"General n-hop over {route} message #{i:08d}" for i in range(message_count)]
        await send_and_receive_packets_with_pop(
            packets,
            src=swarm7[route[0]],
            dest=swarm7[route[-1]],
            path=[swarm7[x]["peer_id"] for x in route[1:-1]],
        )

        await asyncio.wait_for(
            check_unredeemed_tickets_value(swarm7[route[1]], message_count * TICKET_PRICE_PER_HOP), 30.0
        )

        await asyncio.sleep(10)  # wait for aggregation to finish before redeeming
        assert await swarm7[route[1]]["api"].tickets_redeem()

        await asyncio.wait_for(check_all_tickets_redeemed(swarm7[route[1]]), 120.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", [shuffled(default_nodes())[:3] for _ in range(PARAMETERIZED_SAMPLE_SIZE)])
async def test_hoprd_should_be_able_to_close_open_channels_with_unredeemed_tickets(route, swarm7):
    ticket_count = 2

    async with AsyncExitStack() as channels:
        await asyncio.gather(
            *[
                channels.enter_async_context(
                    create_channel(swarm7[route[i]], swarm7[route[i + 1]], funding=ticket_count * TICKET_PRICE_PER_HOP)
                )
                for i in range(len(route) - 1)
            ]
        )

        packets = [f"Channel unredeemed check: #{i:08d}" for i in range(ticket_count)]
        await send_and_receive_packets_with_pop(
            packets, src=swarm7[route[0]], dest=swarm7[route[-1]], path=[swarm7[route[1]]["peer_id"]]
        )

        await asyncio.wait_for(
            check_unredeemed_tickets_value(swarm7[route[1]], ticket_count * TICKET_PRICE_PER_HOP), 30.0
        )

        # NOTE: will be closed on context manager exit


@pytest.mark.asyncio
@pytest.mark.parametrize("src,dest", random_distinct_pairs_from(default_nodes(), count=PARAMETERIZED_SAMPLE_SIZE))
async def test_hoprd_should_be_able_to_open_and_close_channel_without_tickets(src, dest, swarm7):
    async with create_channel(swarm7[src], swarm7[dest], OPEN_CHANNEL_FUNDING_VALUE):
        # the context manager handles opening and closing of the channel with verification
        assert True


@pytest.mark.asyncio
@pytest.mark.parametrize("route", [shuffled(default_nodes())[:3] for _ in range(PARAMETERIZED_SAMPLE_SIZE)])
async def test_hoprd_strategy_automatic_ticket_aggregation_and_redeeming(route, swarm7):
    ticket_count = int(TICKET_AGGREGATION_THRESHOLD)

    async with AsyncExitStack() as channels:
        await asyncio.gather(
            *[
                channels.enter_async_context(
                    create_channel(swarm7[route[i]], swarm7[route[i + 1]], funding=ticket_count * TICKET_PRICE_PER_HOP)
                )
                for i in range(len(route) - 1)
            ]
        )

        statistics_before = await swarm7[route[1]]["api"].get_tickets_statistics()

        packets = [f"Ticket aggregation test: #{i:08d}" for i in range(ticket_count)]
        await send_and_receive_packets_with_pop(
            packets, src=swarm7[route[0]], dest=swarm7[route[-1]], path=[swarm7[route[1]]["peer_id"]]
        )

        async def aggregate_and_redeem_tickets():
            while True:
                statistics_after = await swarm7[route[1]]["api"].get_tickets_statistics()
                redeemed_value = balance_str_to_int(statistics_after.redeemed_value) - balance_str_to_int(
                    statistics_before.redeemed_value
                )
                redeemed_ticket_count = statistics_after.redeemed - statistics_before.redeemed

                if redeemed_value >= AGGREGATED_TICKET_PRICE:
                    break
                else:
                    await asyncio.sleep(0.5)

            assert redeemed_value >= AGGREGATED_TICKET_PRICE
            assert redeemed_ticket_count == pytest.approx(redeemed_value / AGGREGATED_TICKET_PRICE, 0.1)

        await asyncio.wait_for(aggregate_and_redeem_tickets(), 60.0)


@pytest.mark.asyncio
async def test_hoprd_sanity_check_channel_status(swarm7):
    """
    The bash integration-test.sh opens and closes channels that can be visible inside this test scope
    """
    alice_api = swarm7["1"]["api"]

    open_channels = await alice_api.all_channels(include_closed=False)
    open_and_closed_channels = await alice_api.all_channels(include_closed=True)

    assert len(open_and_closed_channels.all) >= len(open_channels.all), "Open and closed channels should be present"

    statuses = [c.status for c in open_and_closed_channels.all]
    assert "Closed" in statuses or "PendingToClose" in statuses, "Closed channels should be present"


@pytest.mark.asyncio
async def test_hoprd_strategy_UNFINISHED():
    """
    ## NOTE: strategy testing will require separate setup so commented out for now until moved
    # test_strategy_setting() {
    #   local node_api="${1}"

    #   settings=$(get_settings ${node_api})
    #   strategy=$(echo ${settings} | jq -r .strategy)
    #   [[ "${strategy}" != "passive" ]] && { msg "Default strategy should be passive, got: ${strategy}"; exit 1; }

    #   channels_count_pre=$(get_all_channels ${node_api} false | jq '.incoming | length')

    #   set_setting ${node_api} "strategy" "promiscuous"

    #   log "Waiting 100 seconds for the node to make connections to other nodes"
    #   sleep 100

    #   channels_count_post=$(get_all_channels ${node_api} false | jq '.incoming | length')
    #   [[ "${channels_count_pre}" -ge "${channels_count_post}" ]] && { msg "Node didn't open any connections by \
    #    itself even when strategy was set to promiscuous: ${channels_count_pre} !>= ${channels_count_post}"; exit 1; }
    #   echo "Strategy setting successfull"
    # }

    # test_strategy_setting ${api4}
    """
    assert True


@pytest.mark.asyncio
async def test_hoprd_check_native_withdraw_results_UNFINISHED():
    """
    # this 2 functions are runned at the end of the tests when withdraw transaction should clear on blockchain
    # and we don't have to block and wait for it
    check_native_withdraw_results() {
    local initial_native_balance="${1}"

    balances=$(api_get_balances ${api1})
    new_native_balance=$(echo ${balances} | jq -r .native)
    [[ "${initial_native_balance}" == "${new_native_balance}" ]] && \
        { msg "Native withdraw failed, pre: ${initial_native_balance}, post: ${new_native_balance}"; exit 1; }

    echo "withdraw native successful"
    }

    # checking statuses of the long running tests
    balances=$(api_get_balances ${api1})
    native_balance=$(echo ${balances} | jq -r .native)
    """
    assert True


@pytest.mark.asyncio
@pytest.mark.parametrize("peer", random.sample(default_nodes(), 1))
async def test_hoprd_check_ticket_price_is_default(peer, swarm7):
    price = await swarm7[peer]["api"].ticket_price()

    assert isinstance(price, int)
    assert price > 0
