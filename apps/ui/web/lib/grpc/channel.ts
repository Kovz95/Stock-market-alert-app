import { createChannel, createClientFactory } from "nice-grpc";
import {
  AlertServiceDefinition,
  type AlertServiceClient,
} from "../../../../../gen/ts/alert/v1/alert";
import {
  DiscordConfigServiceDefinition,
  type DiscordConfigServiceClient,
} from "../../../../../gen/ts/discord/v1/discord";
import {
  PriceServiceDefinition,
  type PriceServiceClient,
} from "../../../../../gen/ts/price/v1/price";
import {
  SchedulerServiceDefinition,
  type SchedulerServiceClient,
} from "../../../../../gen/ts/scheduler/v1/scheduler";

const GRPC_ENDPOINT = process.env.GRPC_ENDPOINT || "127.0.0.1:8081";

const channel = createChannel(GRPC_ENDPOINT, undefined, {
  "grpc.max_receive_message_length": 16 * 1024 * 1024,
  "grpc.max_send_message_length": 16 * 1024 * 1024,
});

const clientFactory = createClientFactory();

export const alertClient: AlertServiceClient = clientFactory.create(
  AlertServiceDefinition,
  channel
);

export const discordClient: DiscordConfigServiceClient = clientFactory.create(
  DiscordConfigServiceDefinition,
  channel
);

export const priceClient: PriceServiceClient = clientFactory.create(
  PriceServiceDefinition,
  channel
);

export const schedulerClient: SchedulerServiceClient = clientFactory.create(
  SchedulerServiceDefinition,
  channel
);
