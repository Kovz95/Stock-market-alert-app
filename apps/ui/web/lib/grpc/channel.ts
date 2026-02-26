import { createChannel, createClientFactory } from "nice-grpc";
import {
  AlertServiceDefinition,
  type AlertServiceClient,
} from "../../../../../gen/ts/alert/v1/alert";
import {
  DiscordConfigServiceDefinition,
  type DiscordConfigServiceClient,
} from "../../../../../gen/ts/discord/v1/discord";

const GRPC_ENDPOINT = process.env.GRPC_ENDPOINT || "localhost:8080";

const channel = createChannel(GRPC_ENDPOINT);

const clientFactory = createClientFactory();

export const alertClient: AlertServiceClient = clientFactory.create(
  AlertServiceDefinition,
  channel
);

export const discordClient: DiscordConfigServiceClient = clientFactory.create(
  DiscordConfigServiceDefinition,
  channel
);
